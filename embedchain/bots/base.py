import logging

from chromadb.errors import InvalidDimensionException
from embedchain.utils.config import read_yaml_file
import importlib
from embedchain.data_formatter import DataFormatter
from embedchain.config.QueryConfig import DOCS_SITE_PROMPT_TEMPLATE
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document


BOT_CONFIG_MAP = {
    "gpt3": {
        "module": "langchain.llms.OpenAI",
        "path": "embedchain/configs/gpt3.yaml",
        "query": "openai.ChatCompletion.create",
    },
    "gpt4": {
        "module": "langchain.llms.OpenAI",
        "path": "embedchain/configs/gpt4.yaml",
        "query": "gpt4all.GPT4All.generate",
    },
}


class Bot:
    def __init__(self, name):
        """
        Initializes the Bot instance, sets up a vector DB client and
        creates a collection.

        :param name: Name of the configuration user wants to use
        """
        if name not in BOT_CONFIG_MAP:
            raise ValueError(f"Bot {name} not found in config map")
        self.name = name
        self.llm = self._load_module(BOT_CONFIG_MAP[name]["module"])
        self.config_path = BOT_CONFIG_MAP[name]["path"]
        self.config = read_yaml_file(self.config_path)
        embedding_fn_config = self.config.get("embedding_model")
        self.embedding_fn = self._load_module(embedding_fn_config["module"])(**embedding_fn_config["config"])
        self.db_client = self.load_db_client()
        self.collection = self.db_client.collection
        self.memory = ConversationBufferMemory()
        self.user_asks = []

    def load_db_client(self):
        # Load the database client based on the config
        db_config = self.config.get("db")
        host, port = db_config.get("host"), db_config.get("port")
        if db_config["name"] == "chroma":
            from embedchain.vectordb.chroma_db import ChromaDB

            db_dir = db_config.get("db_dir")
            host = db_config.get("host")
            port = db_config.get("port")
            # import pdb

            # pdb.set_trace()
            if host and port:
                return ChromaDB(embedding_fn=self.embedding_fn, host=host, port=port)
            else:
                return ChromaDB(embedding_fn=self.embedding_fn, db_dir=db_dir)
        else:
            raise NotImplementedError(f"DB {embedding_fn_config['name']} not implemented")

    def _load_module(self, name):
        try:
            module_components = name.split(".")
            target_name = module_components[-1]
            module_name = ".".join(module_components[:-1])

            module = importlib.import_module(module_name)
            target = getattr(module, target_name)
            return target
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to extract '{target_name}' from module '{module_name}': {e}")

    def _load_and_embed(self, loader, chunker, src, metadata=None):
        """
        Loads the data from the given URL, chunks it, and adds it to database.

        :param loader: The loader to use to load the data.
        :param chunker: The chunker to use to chunk the data.
        :param src: The data to be handled by the loader. Can be a URL for
        remote sources or local content for local loaders.
        :param metadata: Optional. Metadata associated with the data source.
        """
        embeddings_data = chunker.create_chunks(loader, src)
        documents = embeddings_data["documents"]
        metadatas = embeddings_data["metadatas"]
        ids = embeddings_data["ids"]
        where = {"app_id": self.config.get("id")} if self.config.get("id") is not None else {}
        existing_docs = self.collection.get(
            ids=ids,
            where=where,  # optional filter
        )
        existing_ids = set(existing_docs["ids"])

        if len(existing_ids):
            data_dict = {id: (doc, meta) for id, doc, meta in zip(ids, documents, metadatas)}
            data_dict = {id: value for id, value in data_dict.items() if id not in existing_ids}

            if not data_dict:
                print(f"All data from {src} already exists in the database.")
                return

            ids = list(data_dict.keys())
            documents, metadatas = zip(*data_dict.values())

        # Add app id in metadatas so that they can be queried on later
        if self.config.get("id") is not None:
            metadatas = [{**m, "app_id": self.config.get("id")} for m in metadatas]

        # FIXME: Fix the error handling logic when metadatas or metadata is None
        metadatas = metadatas if metadatas else []
        metadata = metadata if metadata else {}
        chunks_before_addition = self.count()

        # Add metadata to each document
        metadatas_with_metadata = [{**meta, **metadata} for meta in metadatas]

        self.collection.add(documents=documents, metadatas=list(metadatas_with_metadata), ids=ids)
        logging.info((f"Successfully saved {src}. New chunks count: " f"{self.count() - chunks_before_addition}"))

    def _format_result(self, results):
        return [
            (Document(page_content=result[0], metadata=result[1] or {}), result[2])
            for result in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def retrieve_from_database(self, input_query, config):
        """
        Queries the vector database based on the given input query.
        Gets relevant doc based on the query

        :param input_query: The query to use.
        :param config: The query configuration.
        :return: The content of the document that matched your query.
        """
        try:
            where = {"app_id": self.config.get("id")} if self.config.get("id") is not None else {}  # optional filter
            result = self.collection.query(
                query_texts=[
                    input_query,
                ],
                n_results=config.number_documents,
                where=where,
            )
        except InvalidDimensionException as e:
            raise InvalidDimensionException(
                e.message()
                + ". This is commonly a side-effect when an embedding function, different from the one used to add the embeddings, is used to retrieve an embedding from the database."  # noqa E501
            ) from None

        results_formatted = self._format_result(result)
        contents = [result[0].page_content for result in results_formatted]
        return contents

    def _append_search_and_context(self, context, web_search_result):
        return f"{context}\nWeb Search Result: {web_search_result}"

    def _generate_prompt(self, input_query, contexts, config, **kwargs):
        """
        Generates a prompt based on the given query and context, ready to be
        passed to an LLM

        :param input_query: The query to use.
        :param contexts: List of similar documents to the query used as context.
        :param config: Optional.
        :return: The prompt
        """
        context_string = (" | ").join(contexts)
        web_search_result = kwargs.get("web_search_result", "")
        if web_search_result:
            context_string = self._append_search_and_context(context_string, web_search_result)
        if not config.history:
            prompt = config.template.substitute(context=context_string, query=input_query)
        else:
            prompt = config.template.substitute(context=context_string, query=input_query, history=config.history)
        return prompt

    def _access_search_and_get_results(self, input_query):
        from langchain.tools import DuckDuckGoSearchRun

        search = DuckDuckGoSearchRun()
        logging.info(f"Access search to get answers for {input_query}")
        return search.run(input_query)

    def add(self, data_type, url, metadata=None, config=None):
        if config is None:
            config = {"chunk_size": 2000, "chunk_overlap": 0, "length_function": len}
        data_formatter = DataFormatter(data_type, config)
        self.user_asks.append([data_type, url, metadata])
        self._load_and_embed(data_formatter.loader, data_formatter.chunker, url, metadata)
        if data_type in ("docs_site",):
            self.is_docs_site_instance = True

    def query(self, input_query, config=None, dry_run=False):
        """
        Queries the vector database based on the given input query.
        Gets relevant doc based on the query and then passes it to an
        LLM as context to get the answer.

        :param input_query: The query to use.
        :param config: Optional. The `QueryConfig` instance to use as
        configuration options.
        :param dry_run: Optional. A dry run does everything except send the resulting prompt to
        the LLM. The purpose is to test the prompt, not the response.
        You can use it to test your prompt, including the context provided
        by the vector database's doc retrieval.
        The only thing the dry run does not consider is the cut-off due to
        the `max_tokens` parameter.
        :return: The answer to the query.
        """
        if config is None:
            config = self.config["query"]
        if self.is_docs_site_instance:
            config.template = DOCS_SITE_PROMPT_TEMPLATE
            config.number_documents = 5
        k = {}
        if self.online:
            k["web_search_result"] = self._access_search_and_get_results(input_query)
        contexts = self.retrieve_from_database(input_query, config)
        prompt = self._generate_prompt(input_query, contexts, config, **k)
        logging.info(f"Prompt: {prompt}")

        if dry_run:
            return prompt

        answer = self.get_answer_from_llm(prompt, config)

        if isinstance(answer, str):
            logging.info(f"Answer: {answer}")
            return answer
        else:
            return self._stream_query_response(answer)

    def _stream_query_response(self, answer):
        streamed_answer = ""
        for chunk in answer:
            streamed_answer = streamed_answer + chunk
            yield chunk
        logging.info(f"Answer: {streamed_answer}")

    def chat(self, input_query, config=None, dry_run=False):
        """
        Queries the vector database on the given input query.
        Gets relevant doc based on the query and then passes it to an
        LLM as context to get the answer.

        Maintains the whole conversation in memory.
        :param input_query: The query to use.
        :param config: Optional. The `ChatConfig` instance to use as
        configuration options.
        :param dry_run: Optional. A dry run does everything except send the resulting prompt to
        the LLM. The purpose is to test the prompt, not the response.
        You can use it to test your prompt, including the context provided
        by the vector database's doc retrieval.
        The only thing the dry run does not consider is the cut-off due to
        the `max_tokens` parameter.
        :return: The answer to the query.
        """
        if config is None:
            config = self.config["chat"]
        if self.is_docs_site_instance:
            config.template = DOCS_SITE_PROMPT_TEMPLATE
            config.number_documents = 5
        k = {}
        if self.online:
            k["web_search_result"] = self._access_search_and_get_results(input_query)
        contexts = self.retrieve_from_database(input_query, config)

        chat_history = self.memory.load_memory_variables({})["history"]

        if chat_history:
            config.set_history(chat_history)

        prompt = self._generate_prompt(input_query, contexts, config, **k)
        logging.info(f"Prompt: {prompt}")

        if dry_run:
            return prompt

        answer = self.get_answer_from_llm(prompt, config)

        self.memory.chat_memory.add_user_message(input_query)

        if isinstance(answer, str):
            self.memory.chat_memory.add_ai_message(answer)
            logging.info(f"Answer: {answer}")
            return answer
        else:
            # this is a streamed response and needs to be handled differently.
            return self._stream_chat_response(answer)

    def _stream_chat_response(self, answer):
        streamed_answer = ""
        for chunk in answer:
            streamed_answer = streamed_answer + chunk
            yield chunk
        self.memory.chat_memory.add_ai_message(streamed_answer)
        logging.info(f"Answer: {streamed_answer}")

    def count(self):
        """
        Count the number of embeddings.

        :return: The number of embeddings.
        """
        return self.collection.count()

    def reset(self):
        """
        Resets the database. Deletes all embeddings irreversibly.
        `App` has to be reinitialized after using this method.
        """
        self.db_client.reset()
