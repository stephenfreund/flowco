import ast
from cmath import phase
import difflib
import json
import re
import textwrap
from typing import Any, Dict, List, Optional

import markdown
from bs4 import BeautifulSoup
from pydantic import BaseModel, ValidationError
from regex import P

from flowco.builder.build import BuildEngine
from flowco.builder.pass_config import PassConfig
from flowco.dataflow.dfg import DataFlowGraph, Edge, Geometry, Node
from flowco.dataflow.phase import Phase
from flowco.util.config import AbstractionLevel
from flowco.util.errors import FlowcoError
from flowco.util.output import log, logger, message
from flowco.util.text import (
    pill_to_function_name,
    pill_to_python_name,
    pill_to_result_var_name,
)
from flowco.util.yes_no import YesNoPrompt


# 1. Define the FlowthonNode BaseModel
class FlowthonNode(BaseModel):
    pill: str  # Function name
    uses: List[str]  # Parameter names
    label: str  # Short description
    requirements: Optional[List[str]] = None
    algorithm: Optional[List[str]] = None

    imports: Optional[List[str]] = None  # New attribute for per-function imports
    code: Optional[List[str]] = None
    assertions: Optional[List[str]] = None

    def update(self, **kwargs) -> "FlowthonNode":
        updated = self.model_copy(update=kwargs)
        if updated == self:
            return self
        else:
            return updated

    def to_json(self, level: AbstractionLevel) -> Dict[str, Any]:
        map = {
            "uses": self.uses,
            "label": self.label,
        }
        if self.requirements:
            map["requirements"] = self.requirements
        if self.algorithm and level in [
            AbstractionLevel.algorithm,
            AbstractionLevel.code,
        ]:
            map["algorithm"] = self.algorithm
        if self.code and level in [AbstractionLevel.code]:
            map["code"] = self.code
        return map

    @classmethod
    def from_json(cls, pill: str, node_data: dict) -> "FlowthonNode":
        assert isinstance(pill, str), f"Expected str, got {type(pill)}"
        assert isinstance(node_data, dict), f"Expected dict, got {type(node_data)}"
        assert all(
            key in ["uses", "label", "requirements", "algorithm", "code", "assertions", "imports"]
            for key in node_data
        ), f"Missing keys in {node_data}"

        # assert that uses is a list of strings and that all strings are valid pills in the nodes
        assert isinstance(
            node_data["uses"], list
        ), f"Expected list, got {type(node_data['uses'])}"
        assert all(
            isinstance(x, str) for x in node_data["uses"]
        ), f"Expected list of str, got {node_data['uses']}"

        # assert label is a string
        assert isinstance(
            node_data["label"], str
        ), f"Expected str, got {type(node_data['label'])}"

        if "requirements" in node_data:
            assert isinstance(
                node_data["requirements"], list
            ), f"Expected list, got {type(node_data['requirements'])}"
            assert all(
                isinstance(x, str) for x in node_data["requirements"]
            ), f"Expected list of str, got {node_data['requirements']}"

        if "algorithm" in node_data:
            assert isinstance(
                node_data["algorithm"], list
            ), f"Expected list, got {type(node_data['algorithm'])}"
            assert all(
                isinstance(x, str) for x in node_data["algorithm"]
            ), f"Expected list of str, got {node_data['algorithm']}"

        if "code" in node_data:
            assert isinstance(
                node_data["code"], list
            ), f"Expected str, got {type(node_data['code'])}"
            assert all(
                isinstance(x, str) for x in node_data["code"]
            ), f"Expected list of str, got {node_data['code']}"
            node_data["code"] = node_data["code"]

        if "assertions" in node_data:
            assert isinstance(
                node_data["assertions"], list
            ), f"Expected list, got {type(node_data['assertions'])}"
            assert all(
                isinstance(x, str) for x in node_data["assertions"]
            ), f"Expected list of str, got {node_data['assertions']}"
            node_data["assertions"] = node_data["assertions"]
 
        if "imports" in node_data:
            assert isinstance(
                node_data["imports"], list
            ), f"Expected list, got {type(node_data['imports'])}"
            assert all(
                isinstance(x, str) for x in node_data["imports"]
            ), f"Expected list of str, got {node_data['imports']}"
            node_data["imports"] = node_data["imports"]

        return cls(
            pill=pill,
            uses=node_data.get("uses", []),
            label=node_data.get("label", ""),
            requirements=node_data.get("requirements", None),
            algorithm=node_data.get("algorithm", None),
            imports=node_data.get("imports", None),
            code=node_data.get("code", None),
            assertions=node_data.get("assertions", None),
        )


class FlowthonProgram(BaseModel):
    tables: List[str] = []
    nodes: Dict[str, FlowthonNode] = {}

    def __init__(self, **data):
        super().__init__(**data)
        self.validate_edges()

    def validate_edges(self):
        # verify that all uses in all nodes are valid pills in the nodes
        for node in self.nodes.values():
            for use in node.uses:
                if use not in self.nodes:
                    raise FlowcoError(f"Node {node.pill} refers to unknown node {use}")

    def update(self, **kwargs) -> "FlowthonProgram":
        updated = self.model_copy(update=kwargs)
        updated.validate_edges()
        if updated == self:
            return self
        else:
            return updated

    def to_json(self, level: AbstractionLevel) -> Dict[str, Any]:
        map = {
            "tables": self.tables,
            "nodes": {pill: node.to_json(level) for pill, node in self.nodes.items()},
        }
        return map

    @classmethod
    def from_json(cls, data: dict) -> "FlowthonProgram":
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"
        assert all(
            key in ["nodes", "tables" ] for key in data
        ), f"Missing keys in {data}"

        if "tables" in data:
            assert isinstance(
                data["tables"], list
            ), f"Expected list, got {type(data['tables'])}"
            assert all(
                isinstance(x, str) for x in data["tables"]
            ), f"Expected list of str, got {data['tables']}"

        if "nodes" in data:
            assert isinstance(
                data["nodes"], dict
            ), f"Expected dict, got {type(data['nodes'])}"
            assert all(
                isinstance(x, str) for x in data["nodes"]
            ), f"Expected dict of str, got {data['nodes']}"

        return cls(
            tables=data.get("tables", []),
            nodes={
                pill: FlowthonNode.from_json(pill, node_data)
                for pill, node_data in data.get("nodes", {}).items()
            },
        )

    @classmethod
    def from_source(cls, data: str) -> "FlowthonProgram":
        # 3. Function to extract function information from source code
        def extract_function_info(source_code: str) -> Dict[str, Dict[str, Any]]:
            """
            Extracts docstrings, parameter names, and source code from all functions in the given source code.

            :param source_code: The Python source code as a string.
            :return: A dictionary mapping function names to their docstrings, parameters, and source code.
            """
            tree = ast.parse(source_code)
            function_info = {}
            lines = source_code.split("\n")
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.append(ast.unparse(node).strip())
                if isinstance(node, ast.FunctionDef):
                    doc = ast.get_docstring(node)
                    params = [arg.arg for arg in node.args.args]

                    # Initialize docstring positions
                    doc_start = None
                    doc_end = None

                    if doc and node.body:
                        first_node = node.body[0]
                        if isinstance(first_node, ast.Expr):
                            value = first_node.value
                            if isinstance(value, (ast.Str, ast.Constant)):
                                doc_start = first_node.lineno - 1  # 0-based index
                                if hasattr(first_node, "end_lineno"):
                                    doc_end = first_node.end_lineno
                                else:
                                    # Fallback: find end of the docstring based on quotes
                                    docstring_line = lines[doc_start]
                                    quote_char = (
                                        '"""' if '"""' in docstring_line else "'''"
                                    )
                                    start_pos = docstring_line.find(quote_char)
                                    end_pos = docstring_line.find(
                                        quote_char, start_pos + len(quote_char)
                                    )
                                    if end_pos == -1:
                                        # Multiline docstring
                                        for i in range(doc_start + 1, len(lines)):
                                            if quote_char in lines[i]:
                                                doc_end = i + 1
                                                break

                    # Extract function code
                    if hasattr(node, "end_lineno"):
                        start = node.lineno
                        end = node.end_lineno
                        func_code = lines[start - 1 : end]
                    else:
                        func_code = extract_function_code_fallback(node, lines).split(
                            "\n"
                        )

                    # Exclude docstring lines
                    if doc_start is not None and doc_end is not None:
                        # Calculate relative indices
                        relative_start = doc_start - (node.lineno - 1)
                        relative_end = doc_end - (node.lineno - 1)
                        # Remove docstring lines
                        func_code = (
                            func_code[:relative_start] + func_code[relative_end:]
                        )
                    
                    if func_code[-1].strip() == "...":
                        func_code = None

                    function_info[node.name] = {
                        "docstring": doc,
                        "parameters": params,
                        "imports": imports,
                        "code": func_code,
                    }

                    imports = []  # Reset imports for the next function

            return function_info

        def extract_function_code_fallback(
            node: ast.FunctionDef, lines: List[str]
        ) -> str:
            """
            Fallback method to extract function code when end_lineno is not available.

            :param node: The AST node representing the function.
            :param lines: List of all lines in the source code.
            :return: The function's source code as a string.
            """
            # Start from the function's starting line
            start = node.lineno - 1  # 0-based index
            # Initialize end as the starting line
            end = start + 1
            # Iterate until we find a line that's not indented more than the function's definition
            func_indent = len(lines[start]) - len(lines[start].lstrip())
            for i in range(start + 1, len(lines)):
                line = lines[i]
                stripped = line.strip()
                if not stripped:
                    continue  # Skip empty lines
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= func_indent:
                    break
                end = i + 1
            return "\n".join(lines[start:end])

        # 4. Function to extract table declarations using regex
        def extract_table_info(line: str) -> Optional[str]:
            """
            Extracts table information from a line that declares a table.

            Expected format: table filename.csv

            :param line: A single line from the input file.
            :return: The filename as a string if the line is a table declaration, else None.
            """
            match = re.match(r"^table\s+([\w\-\.]+\.csv)$", line.strip())
            if match:
                filename = match.group(1)
                return filename
            return None
            
        def parse_markdown_docstring(doc: str, allowed_headers: Dict[str, str]) -> Dict[str, Any]:
            """
            Parses a Markdown-formatted docstring into a structured dictionary based on allowed section headers.
            
            This function manually parses the docstring without converting it to HTML, preserving the original Markdown formatting
            and handling multi-line bullets.
            
            :param doc: The docstring to parse.
            :param allowed_headers: A dictionary mapping FlowthonNode field names to header titles.
            :return: A dictionary with structured data.
            :raises ValueError: If an undefined section header is found.
            """
            sections = {
                "short_description": "",
            }
            
            # Initialize all allowed sections as empty lists
            for field in allowed_headers:
                sections[field] = []
            
            current_section = "short_description"
            buffer = []
            
            # Split the docstring into lines for processing
            lines = doc.splitlines()
            
            # Regular expression to match headers (e.g., # Requirements)
            header_regex = re.compile(r'^(#{1,6})\s+(.*)')
            
            # Regular expression to match bullet points (e.g., - Must handle edge cases)
            bullet_regex = re.compile(r'^-\s+(.*)')
            
            for line in lines:
                stripped_line = line.strip()
                
                # Check if the line is a header
                header_match = header_regex.match(stripped_line)
                if header_match:
                    header_text = header_match.group(2).strip()
                    
                    if header_text not in allowed_headers.values():
                        raise ValueError(f"Undefined section header '{header_text}' found in docstring.")
                    
                    # Map header text to field name (case-insensitive)
                    field_name = None
                    for key, value in allowed_headers.items():
                        if value.lower() == header_text.lower():
                            field_name = key
                            break
                    
                    if not field_name:
                        raise ValueError(f"Header '{header_text}' does not correspond to any FlowthonNode field.")
                    
                    # Assign buffered content to the previous section
                    if buffer:
                        if current_section == "short_description":
                            # Concatenate all buffered lines for short description
                            sections["short_description"] = ' '.join(buffer).strip()
                        elif current_section in sections and isinstance(sections[current_section], list):
                            # Process buffered lines to extract bullet points, including multi-line bullets
                            bullets = []
                            current_bullet = ""
                            for buf_line in buffer:
                                bullet_match = bullet_regex.match(buf_line)
                                if bullet_match:
                                    if current_bullet:
                                        bullets.append(current_bullet.strip())
                                    current_bullet = bullet_match.group(1)
                                elif buf_line.startswith('    ') or buf_line.startswith('\t'):
                                    # Continuation of the current bullet (indented line)
                                    current_bullet += ' ' + buf_line.strip()
                                else:
                                    # Non-indented line without a bullet; append to current bullet
                                    current_bullet += ' ' + buf_line.strip()
                            if current_bullet:
                                bullets.append(current_bullet.strip())
                            
                            sections[current_section].extend(bullets)
                        
                        buffer = []
                    
                    # Update the current section
                    current_section = field_name
                    continue  # Move to the next line
                
                # Accumulate lines for the current section
                if stripped_line:
                    buffer.append(stripped_line)
                else:
                    # Blank line indicates possible separation between paragraphs or sections
                    if current_section == "short_description":
                        if buffer:
                            sections["short_description"] += ' ' + ' '.join(buffer).strip()
                            buffer = []
                    else:
                        # For list sections, blank lines within bullets are treated as continuation
                        if buffer:
                            buffer.append(stripped_line)
            
            # After processing all lines, assign any remaining buffer content
            if buffer:
                if current_section == "short_description":
                    sections["short_description"] = buffer[0].strip()
                elif current_section in sections and isinstance(sections[current_section], list):
                    bullets = []
                    current_bullet = ""
                    for buf_line in buffer:
                        bullet_match = bullet_regex.match(buf_line)
                        if bullet_match:
                            if current_bullet:
                                bullets.append(current_bullet.strip())
                            current_bullet = bullet_match.group(1)
                        elif buf_line.startswith('    ') or buf_line.startswith('\t'):
                            # Continuation of the current bullet (indented line)
                            current_bullet += ' ' + buf_line.strip()
                        else:
                            # Non-indented line without a bullet; append to current bullet
                            current_bullet += ' ' + buf_line.strip()
                    if current_bullet:
                        bullets.append(current_bullet.strip())
                    
                    sections[current_section].extend(bullets)
            
            return sections


        # 6. Function to map extracted data to FlowthonNode
        def map_to_flowthon_node(
            func_name: str, info: Dict[str, Any], sections: Dict[str, Any]
        ) -> Optional[FlowthonNode]:
            """
            Maps extracted function information and parsed sections to a FlowthonNode instance.

            :param func_name: Name of the function.
            :param info: Dictionary containing 'parameters' and 'code' of the function.
            :param sections: Dictionary containing parsed docstring sections.
            :return: A FlowthonNode instance or None if validation fails.
            """
            code = info.get("code")
            if code:
                code = info.get("imports", []) + code
                
            try:
                node = FlowthonNode(
                    pill=func_name,
                    uses=info.get("parameters", []),
                    label=sections.get("short_description", ""),  # Wrap in list
                    requirements=sections.get("requirements"),
                    algorithm=sections.get("algorithm"),
                    code=code,
                    assertions=sections.get("assertions"),
                    preconditions=sections.get("preconditions"),
                    steps=sections.get("steps"),
                )
                return node
            except ValidationError as e:
                print(f"Validation error for function '{func_name}': {e}")
                return None

        """
        Parses the input text and converts each import, table declaration, and function into respective BaseModel instances.

        :param input_text: The content of the input file as a string.
        :return: A ParsedFile instance containing tables, and functions.
        :raises ValueError: If an undefined section header is found in any docstring.
        """
        table_filenames = []
        python_code_lines = []

        for line in data.split("\n"):
            stripped = line.strip()
            # if stripped.startswith("import ") or stripped.startswith("from "):
            #     import_lines.append(line)
            if stripped.startswith("table "):
                table_filename = extract_table_info(line)
                if table_filename:
                    table_filenames.append(table_filename)
            else:
                python_code_lines.append(line)

        # Parse table declarations as list of filenames
        tables = table_filenames

        # Parse Python code for functions
        python_code = "\n".join(python_code_lines)
        functions = {}
        errors = []
        if python_code.strip():
            function_infos = extract_function_info(python_code)
            if function_infos:
                # Dynamically extract allowed section headers from FlowthonNode
                allowed_headers = {}
                for field in FlowthonNode.model_fields.keys():
                    if field in ["pill", "uses", "label"]:
                        continue  # These are not section headers
                    # Convert field name to title case with spaces
                    header_title = field.replace("_", " ").title()
                    allowed_headers[field] = header_title

                try:
                    tree = ast.parse(python_code)
                    for node in tree.body:
                        if isinstance(node, ast.FunctionDef):
                            func_name = node.name
                            if func_name in function_infos:
                                info = function_infos[func_name]
                                try:
                                    sections = parse_markdown_docstring(
                                        info["docstring"], allowed_headers
                                    )
                                    flowthon_node = map_to_flowthon_node(
                                        func_name, info, sections
                                    )
                                    if flowthon_node:
                                        functions[func_name] = flowthon_node
                                except ValueError as ve:
                                    errors += [f"SyntaxError in `{func_name}`: {ve}"]
                except SyntaxError as se:
                    errors += [f"SyntaxError in `{func_name}`: {se}"]

        if errors:
            raise FlowcoError("\n".join(errors))

        parsed_file = FlowthonProgram(tables=tables, nodes=functions)
        return parsed_file

    def to_source(
        self, abstraction_level: AbstractionLevel = AbstractionLevel.spec
    ) -> str:
        lines = []

        # 2. Add table declarations
        for table in self.tables:
            lines.append(f"table {table}")
        if self.tables:
            lines.append("")  # Add a blank line after tables

        # 3. Add function definitions
        for func in self.nodes.values():
 
            print(abstraction_level)
            print(func.pill)
            print(func.code)

            headers = []
            if abstraction_level == AbstractionLevel.code:
                if func.imports:
                    headers += func.imports


            if func.code:
                # get all lines up to and including the first one that ends with :
                code = func.code
                first_line = next(
                    (line for line in code if line.strip().endswith(":")), None
                )
                assert first_line.startswith("def")
                first_line_index = code.index(first_line)
                headers += code[:code.index(first_line)+1]
                body = code[first_line_index + 1 :]
            else:
                params = ", ".join(func.uses)
                headers += [f"def {func.pill}({params}):"]
                body = ["    ..."]

            lines.extend(headers)

            # Start docstring
            lines.append('    """')

            # Short description (label)
            if func.label:
                lines.append(f"    {func.label}")
                lines.append("")

            # Define the order of sections based on the order of fields in FlowthonNode
            # Exclude fields that are not documentation sections
            section_order = [
                field
                for field in FlowthonNode.model_fields.keys()
                if field not in ["pill", "uses", "label", "code"]
            ]

            for section in section_order:
                content = getattr(func, section)

                if section == "code" and abstraction_level != AbstractionLevel.code:
                    continue

                if section == "algorithm" and (
                    abstraction_level != AbstractionLevel.algorithm
                    and abstraction_level != AbstractionLevel.code
                ):
                    continue

                if content:
                    # Capitalize the first letter of the section name for the header
                    header = section.capitalize()
                    lines.append(f"    # {header}")
                    for item in content:
                        if item.startswith("- "):
                            lines.append(f"      {item}")
                        else:
                            lines.append(f"    - {item}")
                    lines.append("")  # Add a blank line after each section

            # Remove the last blank line if it exists to avoid extra newline before closing docstring
            if lines[-1] == "":
                lines.pop()

            # End docstring
            lines.append('    """')

            # Function implementation code
            if abstraction_level == AbstractionLevel.code and body:
                for line in body:
                    lines.append(f"{line}")
            else:
                lines.append("    ...")

            lines.append("")  # Add a blank line after each function

        # Remove the last blank line to avoid trailing newline
        if lines and lines[-1] == "":
            lines.pop()

        # Combine all lines into a single string with newline characters
        return "\n".join(lines)

    @classmethod
    def from_file(cls, file_name: str) -> "FlowthonProgram":
        try:
            with open(file_name, "r") as file:
                data = json.load(file)
            return cls.from_json(data)
        except FileNotFoundError:
            raise FlowcoError(f"File {file_name} does not exist.")
        except json.JSONDecodeError:
            log(f"File {file_name} is not a valid JSON file.")
            try:
                with open(file_name, "r") as file:
                    data = file.read()
                return cls.from_source(data)
            except FlowcoError as e:
                raise e

    def merge_code(
        self, current: List[str] | None, incoming: List[str] | None
    ) -> List[str] | None:
        if current is None:
            return incoming
        if incoming is None:
            return current

        print(current)
        print(incoming)

        current_header = next(
            (line for line in current if line.strip().endswith(":")), None
        )
        incoming_header = next(
            (line for line in incoming if line.strip().endswith(":")), None
        )

        current_header_index = current.index(current_header) if current_header else None
        incoming_header_index = (
            incoming.index(incoming_header) if incoming_header else None
        )

        return (
            current[: current_header_index + 1] + incoming[incoming_header_index + 1 :]
        )

    def merge(
        self,
        pass_config: PassConfig,
        dfg: DataFlowGraph | None = None,
        interactive=False,
    ) -> DataFlowGraph:
        """
        Merge the editable graph into the DataFlowGraph:
        - Add new nodes
        - Update existing nodes
        - Remove nodes that are not in the editable graph
        - Update edges
        """
        if dfg is None:
            dfg = DataFlowGraph(nodes=[], edges=[], version=0)

        new_nodes = []
        new_edges = []

        for node in self.nodes.values():
            if node.pill not in dfg.node_ids():
                new_node = Node(
                    id=node.pill,
                    pill=node.pill,
                    label=node.label,
                    phase=Phase.clean,
                    predecessors=[],
                    geometry=Geometry(x=0, y=0, width=0, height=0),
                    output_geometry=Geometry(x=0, y=0, width=0, height=0),
                    function_name=pill_to_function_name(node.pill),
                    function_result_var=pill_to_result_var_name(node.pill),
                    requirements=node.requirements,
                    algorithm=node.algorithm,
                    code=node.code,
                    assertions=node.assertions,
                )

                message(f"Adding new node {new_node.pill}")

            else:
                original = dfg[node.pill]
                new_node = original.model_copy(
                    update={
                        "predecessors": [],
                        "phase": Phase.clean,
                    }
                )

                new_node = new_node.update(
                    label=node.label,
                    requirements=(
                        node.requirements
                        if node.requirements is not None
                        else new_node.requirements
                    ),
                    algorithm=(
                        node.algorithm
                        if node.algorithm is not None
                        else new_node.algorithm
                    ),
                    code=(
                        node.code
                        if node.code is not None
                        else new_node.code
                    )
                    
#                    self.merge_code(new_node.code, node.code),
                )

                edits = []
                if node.label != original.label:
                    edits.append("label")
                if node.requirements and node.requirements != original.requirements:
                    edits.append("requirements")
                if node.algorithm and node.algorithm != original.algorithm:
                    edits.append("algorithm")
                if node.code and node.code != original.code:
                    edits.append("code")
                if node.assertions and node.assertions != original.assertions:
                    edits.append("assertions")

                if edits:
                    message(
                        f"Modifying existing node {new_node.pill}: {', '.join(edits)}"
                    )

            new_nodes.append(new_node)
            for pred in node.uses:
                new_edges.append(
                    Edge(id=f"{pred}->{node.pill}", src=pred, dst=node.pill)
                )

        new_graph = DataFlowGraph(
            description=dfg.description,
            nodes=new_nodes,
            edges=new_edges,
            version=dfg.version + 1,
        )

        def predecessors(node: Node) -> List[str]:
            direct_preds = {edge.src for edge in new_graph.edges if edge.dst == node.id}
            indirect_preds = {
                p for pred in direct_preds for p in predecessors(new_graph[pred])
            }
            preds = list(direct_preds | indirect_preds)
            preds = sorted(preds, key=lambda x: new_graph[x].pill)
            return preds

        for node_id in new_graph.topological_sort():
            new_graph[node_id].predecessors = predecessors(new_graph[node_id])
            if (
                node_id in dfg
                and new_graph[node_id].predecessors != dfg[node_id].predecessors
            ):
                message(
                    f"Modifying existing node {new_graph[node_id].pill}: predecessors"
                )

        new_graph = self.rebuild(pass_config, dfg, new_graph, interactive)

        return new_graph

    def rebuild(
        self,
        pass_config: PassConfig,
        original: DataFlowGraph,
        dfg: DataFlowGraph,
        interactive,
    ) -> DataFlowGraph:

        with logger("Making build engine"):
            engine = BuildEngine.get_builder()

            attributes = {
                Phase.requirements: "requirements",
                Phase.algorithm: "algorithm",
                Phase.code: "code",
            }

            yes_no_prompt = YesNoPrompt(not interactive)

            with logger("Building"):
                for build_updated in engine.build_with_worklist(
                    pass_config, dfg, Phase.run_checked, None
                ):
                    new_dfg = build_updated.new_graph
                    updated_node = build_updated.updated_node

                    if updated_node is not None:
                        prefix = f"[{str(updated_node.phase)}]"
                        prefix = f"{prefix:<14} {updated_node.pill}"
                        prefix = f" {prefix:.<55}"

                        if updated_node.phase in attributes:
                            key = attributes[updated_node.phase]
                            edited = getattr(self.nodes[updated_node.id], key)
                            new = getattr(updated_node, key)

                            if updated_node.id in original.node_ids():
                                old = getattr(original[updated_node.id], key)
                            else:
                                old = None

                            if old == new:
                                message(f"{prefix} unchanged")
                            elif edited is None:
                                message(f"{prefix} generated")
                            elif edited == new:
                                message(f"{prefix} taken from user")
                            else:
                                message(f"{prefix} taken from user and modified")
                                diff = "\n".join(difflib.ndiff(edited, new))
                                print(textwrap.indent(diff, "    "))
                                print()
                                if yes_no_prompt.ask("Accept this change"):
                                    setattr(self.nodes[updated_node.id], key, new)
                                    print()
                                else:
                                    print()
                                    raise FlowcoError("User rejected changes")

                    else:
                        message(f"{prefix} done")

            with logger("Updating Build Caches"):
                updated = [
                    node.update(cache=node.cache.update_all(node))
                    for node in new_dfg.nodes
                ]
                new_dfg = new_dfg.update(nodes=updated)

            return new_dfg
