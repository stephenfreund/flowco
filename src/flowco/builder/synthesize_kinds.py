from flowco.assistant.flowco_assistant import flowco_assistant
from flowco.builder.synthesize import requirements, compile
from flowco.dataflow.extended_type import ExtendedType
from flowco.dataflow.phase import Phase
from flowco.builder.build import PassConfig, node_pass

from flowco.dataflow.dfg import DataFlowGraph, Node, NodeKind
from flowco.dataflow.preconditions import FunctionPreconditions
from flowco.page.tables import file_path_to_table_name, table_df
from flowco.util.config import config
from flowco.util.output import logger


def table_requirements(
    pass_config: PassConfig, graph: DataFlowGraph, node: Node
) -> Node:
    with logger("table_requirements"):
        name = node.label.split("`")[1]
        df = table_df(name)
        function_return_type = ExtendedType.from_value(df)
        function_return_type.description += f"The DataFrame for the {name} dataset."  #  Here are the first few rows:\n```\n{df.head()}\n```\n"
        requirements = [
            f"The result is the dataframe for the `{node.pill}` dataset.",
        ]

        assistant = flowco_assistant("system-prompt")
        prompt = config().get_prompt(
            "load-table-return-type",
            table=str(df.head()),
            return_type=function_return_type.model_dump_json(indent=2),
        )
        assistant.add_text("user", prompt)
        completion = assistant.model_completion(ExtendedType)
        print(completion)
        return node.update(
            function_parameters=[],
            preconditions=FunctionPreconditions(),
            function_return_type=completion,
            requirements=requirements,
            function_computed_value="",
            description="",
            phase=Phase.requirements,
        )


template = """
def {function_name}() -> pd.DataFrame:
    df = {table_name}_table()
    # Replace missing values in all string columns with 'Unknown'
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].fillna('Unknown')
    return df
"""


def table_compile(pass_config: PassConfig, graph: DataFlowGraph, node: Node) -> Node:
    with logger("table_compile"):
        name = file_path_to_table_name(node.label.split("`")[1])
        if name.endswith(".csv"):
            code = template.format(
                function_name=node.function_name,
                table_name=name,
            )
        else:
            code = template.format(
                function_name=node.function_name,
                table_name=name,
            )
        return node.update(
            code=code.splitlines(),
            phase=Phase.code,
        )


@node_pass(
    required_phase=Phase.clean,
    target_phase=Phase.requirements,
    pred_required_phase=Phase.requirements,
)
def kind_requirements(
    pass_config: PassConfig, graph: DataFlowGraph, node: Node
) -> Node:
    if node.kind is NodeKind.table:
        return table_requirements(pass_config, graph, node)
    else:
        return requirements(pass_config, graph, node)


@node_pass(required_phase=Phase.algorithm, target_phase=Phase.code)
def kind_compile(pass_config: PassConfig, graph: DataFlowGraph, node: Node) -> Node:
    if node.kind is NodeKind.table:
        return table_compile(pass_config, graph, node)
    else:
        return compile(pass_config, graph, node)
