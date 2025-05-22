"""Question answering over a graph."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    PromptTemplate,
    BasePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import Runnable
from neo4j.exceptions import Neo4jError

from langchain_neo4j.chains.graph_qa.cypher_utils import (
    CypherQueryCorrector,
    Schema,
)
from langchain_neo4j.chains.graph_qa.cypher import (
    construct_schema,
    get_function_response
)
from langchain_neo4j.chains.graph_qa.prompts import (
    CYPHER_GENERATION_PROMPT,
    CYPHER_QA_PROMPT,
)
from langchain_neo4j import GraphCypherQAChain


INTERMEDIATE_STEPS_KEY = "context"
GENERATED_CYPHER_KEY = "cypher"

FUNCTION_RESPONSE_SYSTEM = """You are an assistant that helps to form nice and human 
understandable answers based on the provided information from tools.
Do not add any other information that wasn't present in the tools, and use 
very concise style in interpreting results!
"""

CYPHER_FIX_TEMPLATE = """Task: Address the Neo4j Cypher query error message.

You are a Neo4j Cypher query expert responsible for correcting the provided `Generated Cypher Query`
based on the provided `Cypher Error`.

The `Cypher Error` explains why the `Generated Cypher Query` could not be executed in the database.

Generated Cypher Query:
{cypher_query}

Cypher Error:
{cypher_error}

Instructions:
- Use only the provided relationship types and properties in the schema.
- Do not use any other relationship types or properties that are not provided.

Neo4j Schema:
{schema}

Note:
- Do not include any explanations or apologies in your responses.
- Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
- Do not include any text except the generated Cypher statement.

You will output the `Corrected Cypher Query` wrapped in 3 backticks (```).
Do not include any text except the `Corrected Cypher Query`.

Remember to think step by step.

Corrected Cypher Query:
"""

CYPHER_FIX_PROMPT = PromptTemplate(
    input_variables=["schema", "cypher_query", "cypher_error"], template=CYPHER_FIX_TEMPLATE
)


def extract_cypher(text: str) -> str:
    """Extract and format Cypher query from text, handling code blocks and special characters.

    This function performs two main operations:
    1. Extracts Cypher code from within triple backticks (```), if present
    2. Automatically adds backtick quotes around multi-word identifiers:
       - Node labels (e.g., ":Data Science" becomes ":`Data Science`")
       - Property keys (e.g., "first name:" becomes "`first name`:")
       - Relationship types (e.g., "[:WORKS WITH]" becomes "[:`WORKS WITH`]")

    Args:
        text (str): Raw text that may contain Cypher code, either within triple
                   backticks or as plain text.

    Returns:
        str: Properly formatted Cypher query with correct backtick quoting.
    """
    # Extract Cypher code enclosed in triple backticks
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    cypher_query = matches[0] if matches else text
    
    ############################################################################
    matches = re.search(r"MATCH|CALL|RETURN|YIELD", cypher_query, re.IGNORECASE)
    cypher_query = cypher_query if matches else ""
    ############################################################################

    # Quote node labels in backticks if they contain spaces and are not already quoted
    cypher_query = re.sub(
        r":\s*(?!`\s*)(\s*)([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)+)(?!\s*`)(\s*)",
        r":`\2`",
        cypher_query,
    )
    # Quote property keys in backticks if they contain spaces and are not already quoted
    cypher_query = re.sub(
        r"([,{]\s*)(?!`)([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)+)(?!`)(\s*:)",
        r"\1`\2`\3",
        cypher_query,
    )
    # Quote relationship types in backticks if they contain spaces and are not already quoted
    cypher_query = re.sub(
        r"(\[\s*[a-zA-Z0-9_]*\s*:\s*)(?!`)([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)+)(?!`)(\s*(?:\]|-))",
        r"\1`\2`\3",
        cypher_query,
    )
    return cypher_query


class GraphCypherQAChainMod(GraphCypherQAChain):
    """Chain for question-answering against a graph by generating Cypher statements.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    ###############################################
    cypher_fix_chain: Runnable[Dict[str, Any], str]
    example_key: str = "example"  #: :meta private:
    max_cypher_generation_attempts: int = 3
    """The maximum amount of Cypher Generation attempts that should be made"""
    ###############################################

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the chain."""
        super().__init__(**kwargs)
        if self.allow_dangerous_requests is not True:
            raise ValueError(
                "In order to use this chain, you must acknowledge that it can make "
                "dangerous requests by setting `allow_dangerous_requests` to `True`."
                "You must narrowly scope the permissions of the database connection "
                "to only include necessary permissions. Failure to do so may result "
                "in data corruption or loss or reading sensitive data if such data is "
                "present in the database."
                "Only use this chain if you understand the risks and have taken the "
                "necessary precautions. "
                "See https://python.langchain.com/docs/security for more information."
            )
    
    @property
    def example_keys(self) -> List[str]:
        """Return the example keys.

        :meta private:
        """
        return [self.example_key]

    @classmethod
    def from_llm(
        cls,
        llm: Optional[BaseLanguageModel] = None,
        *,
        qa_prompt: Optional[BasePromptTemplate] = None,
        cypher_generation_prompt: Optional[BasePromptTemplate] = None,
        #############################################
        cypher_fix_prompt: Optional[BasePromptTemplate] = None,
        #############################################
        cypher_llm: Optional[BaseLanguageModel] = None,
        qa_llm: Optional[BaseLanguageModel] = None,
        exclude_types: List[str] = [],
        include_types: List[str] = [],
        validate_cypher: bool = False,
        qa_llm_kwargs: Optional[Dict[str, Any]] = None,
        cypher_llm_kwargs: Optional[Dict[str, Any]] = None,
        use_function_response: bool = False,
        function_response_system: str = FUNCTION_RESPONSE_SYSTEM,
        **kwargs: Any,
    ) -> GraphCypherQAChain:
        """Initialize from LLM."""
        # Ensure at least one LLM is provided
        if llm is None and qa_llm is None and cypher_llm is None:
            raise ValueError("At least one LLM must be provided")

        # Prevent all three LLMs from being provided simultaneously
        if llm is not None and qa_llm is not None and cypher_llm is not None:
            raise ValueError(
                "You can specify up to two of 'cypher_llm', 'qa_llm'"
                ", and 'llm', but not all three simultaneously."
            )

        # Assign default LLMs if specific ones are not provided
        if llm is not None:
            qa_llm = qa_llm or llm
            cypher_llm = cypher_llm or llm
        else:
            # If llm is None, both qa_llm and cypher_llm must be provided
            if qa_llm is None or cypher_llm is None:
                raise ValueError(
                    "If `llm` is not provided, both `qa_llm` and `cypher_llm` must be "
                    "provided."
                )
        if cypher_generation_prompt:
            if cypher_llm_kwargs:
                raise ValueError(
                    "Specifying cypher_generation_prompt and cypher_llm_kwargs together is"
                    " not allowed. Please pass generation_prompt via cypher_llm_kwargs."
                )
        else:
            if cypher_llm_kwargs:
                cypher_generation_prompt = cypher_llm_kwargs.pop(
                    "generation_prompt", CYPHER_GENERATION_PROMPT
                )
                if not isinstance(cypher_generation_prompt, BasePromptTemplate):
                    raise ValueError(
                        "The cypher_llm_kwargs `generation_prompt` must inherit from "
                        "BasePromptTemplate"
                    )
            else:
                cypher_generation_prompt = CYPHER_GENERATION_PROMPT
        ############################################################################
        if cypher_fix_prompt:
            if cypher_llm_kwargs:
                raise ValueError(
                    "Specifying cypher_fix_prompt and cypher_llm_kwargs together is"
                    " not allowed. Please pass fix_prompt via cypher_llm_kwargs."
                )
        else:
            if cypher_llm_kwargs:
                cypher_fix_prompt = cypher_llm_kwargs.pop(
                    "fix_prompt", CYPHER_FIX_PROMPT
                )
                if not isinstance(cypher_fix_prompt, BasePromptTemplate):
                    raise ValueError(
                        "The cypher_llm_kwargs `fix_prompt` must inherit from "
                        "BasePromptTemplate"
                    )
            else:
                cypher_fix_prompt = CYPHER_FIX_PROMPT
        ############################################################################
        if qa_prompt:
            if qa_llm_kwargs:
                raise ValueError(
                    "Specifying qa_prompt and qa_llm_kwargs together is"
                    " not allowed. Please pass prompt via qa_llm_kwargs."
                )
        else:
            if qa_llm_kwargs:
                qa_prompt = qa_llm_kwargs.pop("prompt", CYPHER_QA_PROMPT)
                if not isinstance(qa_prompt, BasePromptTemplate):
                    raise ValueError(
                        "The qa_llm_kwargs `prompt` must inherit from "
                        "BasePromptTemplate"
                    )
            else:
                qa_prompt = CYPHER_QA_PROMPT
        use_qa_llm_kwargs = qa_llm_kwargs if qa_llm_kwargs is not None else {}
        use_cypher_llm_kwargs = (
            cypher_llm_kwargs if cypher_llm_kwargs is not None else {}
        )

        if use_function_response:
            try:
                if hasattr(qa_llm, "bind_tools"):
                    qa_llm.bind_tools({})
                else:
                    raise AttributeError
                response_prompt = ChatPromptTemplate.from_messages(
                    [
                        SystemMessage(content=function_response_system),
                        HumanMessagePromptTemplate.from_template("{question}"),
                        MessagesPlaceholder(variable_name="function_response"),
                    ]
                )
                qa_chain = response_prompt | qa_llm
            except (NotImplementedError, AttributeError):
                raise ValueError("Provided LLM does not support native tools/functions")
        #######################################################################
        else:
            qa_chain = qa_prompt | qa_llm.bind(**use_qa_llm_kwargs)
        cypher_generation_chain = (
            cypher_generation_prompt | cypher_llm.bind(**use_cypher_llm_kwargs)
        )
        cypher_fix_chain = (
            cypher_fix_prompt | cypher_llm.bind(**use_cypher_llm_kwargs)
        )
        #######################################################################

        if exclude_types and include_types:
            raise ValueError(
                "Either `exclude_types` or `include_types` "
                "can be provided, but not both"
            )
        graph = kwargs["graph"]
        graph_schema = construct_schema(
            graph.get_structured_schema,
            include_types,
            exclude_types,
            graph._enhanced_schema,
        )

        cypher_query_corrector = None
        if validate_cypher:
            corrector_schema = [
                Schema(el["start"], el["type"], el["end"])
                for el in graph.get_structured_schema.get("relationships", [])
            ]
            cypher_query_corrector = CypherQueryCorrector(corrector_schema)

        return cls(
            graph_schema=graph_schema,
            qa_chain=qa_chain,
            cypher_generation_chain=cypher_generation_chain,
            ##################################
            cypher_fix_chain=cypher_fix_chain,
            ##################################
            cypher_query_corrector=cypher_query_corrector,
            use_function_response=use_function_response,
            **kwargs,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Generate Cypher statement, use it to look up in db and answer question."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        question = inputs[self.input_key]
        example = inputs.get(self.example_key, "")
        args = {
            "question": question,
            "example": example,
            "schema": self.graph_schema,
        }
        args.update(inputs)

        list_generated_cypher: List = []

        #####################################
        str_output_parser = StrOutputParser()
        #####################################

        generated_cypher = self.cypher_generation_chain.invoke(
            args, callbacks=callbacks
        )
        list_generated_cypher.append(generated_cypher)

        cypher_query = ""
        cypher_error = ""
        cypher_result = None
        cypher_generation_attempt = 1

        while (
            cypher_result is None
            and cypher_generation_attempt < self.max_cypher_generation_attempts + 1
        ):
            #########################################################################
            # Extract Cypher code if it is wrapped in backticks
            cypher_query = extract_cypher(str_output_parser.invoke(generated_cypher))
            #########################################################################

            # Correct Cypher query if enabled
            if self.cypher_query_corrector:
                cypher_query = self.cypher_query_corrector(cypher_query)

            _run_manager.on_text(
                f"Generated Cypher ({cypher_generation_attempt}):", end="\n",
                verbose=self.verbose
            )
            _run_manager.on_text(
                cypher_query.strip(), color="green", end="\n\n", verbose=self.verbose
            )

            # Retrieve and limit the number of results
            # Generated Cypher be null if query corrector identifies invalid schema
            if cypher_query:
                ######################################################################
                try:
                    cypher_result = self.graph.query(cypher_query)[: self.top_k]
                    if not cypher_result:
                        cypher_result = []
                except Neo4jError as e:
                    cypher_error = f"{e.code}\n{e.message}"

                    _run_manager.on_text(
                    "Cypher Query Execution Error: ", end="\n", verbose=self.verbose
                    )
                    _run_manager.on_text(
                        cypher_error.strip(), color="yellow", end="\n\n", verbose=self.verbose
                    )
                    
                    # Retry Cypher Generation
                    generated_cypher = self.cypher_fix_chain.invoke(
                        {
                            "schema": self.graph_schema,
                            "cypher_query": cypher_query,
                            "cypher_error": cypher_error
                        },
                        callbacks=callbacks,
                    )

                    list_generated_cypher.append(generated_cypher)
                    cypher_generation_attempt += 1
                ######################################################################
            else:
                cypher_result = []

        ############################################################################
        _run_manager.on_text("Skip QA LLM:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            str(self.return_direct), color="green", end="\n\n", verbose=self.verbose
        )
        ############################################################################
        
        _run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            str(cypher_result), color="green", end="\n\n", verbose=self.verbose
        )

        final_result: Union[List[Dict[str, Any]], str]
        ###########################################
        if self.return_direct or not cypher_result:
        ###########################################
            final_result = cypher_result
        else:

            if self.use_function_response:
                function_response = get_function_response(question, cypher_result)
                final_result = self.qa_chain.invoke(
                    {"question": question, "function_response": function_response},
                )
            else:
                final_result = self.qa_chain.invoke(
                    {"question": question, "context": cypher_result},
                    callbacks=callbacks,
                )

        #######################################################################
        chain_result: Dict[str, Any] = {GENERATED_CYPHER_KEY: list_generated_cypher}
        if self.return_intermediate_steps:
            chain_result[INTERMEDIATE_STEPS_KEY] = cypher_result
        chain_result[self.output_key] = final_result
        #######################################################################

        return chain_result
