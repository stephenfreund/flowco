import { Streamlit, RenderData } from "streamlit-component-lib";
import mx from './mxgraph';
import { getCompletion } from "./completion";
import { mxGraphModel, mxUtils, mxEvent, mxRectangle, mxCellState, mxVertexHandler, mxEdgeHandler, mxCell, mxCellEditor, mxHierarchicalLayout } from 'mxgraph';
import { v4 as uuidv4 } from "uuid";
import * as _ from "lodash";

import { DiagramNode, DiagramEdge, mxDiagram, updateDiagram, convertMxGraphToDiagramUpdate, NodeValue, node_style, labelForEdge, toSnakeCase, node_hover_style, clean_color, isDiagramNode } from "./diagram";

var currentDiagram: mxDiagram | undefined = undefined;
var currentRefreshPhase = 0;

// Create a container for the graph
const graphContainer = document.querySelector("#graph-container") as HTMLDivElement;
const graph = new mx.mxGraph(graphContainer);


var can_edit = true;
graph.setPanning(true);
graph.setHtmlLabels(true);

graph.setAllowDanglingEdges(false);
graph.setDisconnectOnMove(false);

graph.setConnectable(true);
graph.setCellsSelectable(true);
graph.setCellsEditable(true);
graph.setCellsResizable(true);
graph.setEnterStopsCellEditing(true);


function setEditable(editable: boolean) {
  can_edit = editable;
  graph.setEnabled(can_edit);
}

graphContainer.addEventListener('contextmenu', (event) => {
  event.preventDefault();
});

graphContainer.addEventListener('click', (event) => {
  // Set focus back to the graph container
  graphContainer.focus();
});

// Ensure the graph container is focusable
graphContainer.setAttribute('tabindex', '0');


// Class representing a set of icons displayed on vertex hover
class mxIconSet {
  private images: HTMLImageElement[] | null;

  constructor(private state: mxCellState) {
    this.images = [];
    const graph = state.view.graph;

    if (state.cell.id.startsWith("output")) {
      return;
    } else if (state.cell.isVertex()) {
      // tset if the current cell and all predecessors have phase 1 or better.
      const upToDate = (cell: mxCell): boolean => {
        if (cell.value.phase < currentRefreshPhase) {
          console.log('not up to date', cell.id, cell.value.phase, currentRefreshPhase)
          return false;
        }
        const incomingEdges = graph.getIncomingEdges(cell, graph.getDefaultParent());
        for (const edge of incomingEdges) {
          const source = edge.source;
          if (!upToDate(source)) {
            return false;
          }
        }
        console.log('up to date', cell.id, cell.value.phase, currentRefreshPhase)
        return true;
      }

      const up_to_date = upToDate(state.cell);
      const editImg: HTMLImageElement = mx.mxUtils.createImage(up_to_date ? 'edit.png' : 'refresh.png');
      editImg.setAttribute('title', up_to_date ? 'Edit' : 'Refresh');
      Object.assign(editImg.style, {
        position: 'absolute',
        cursor: 'pointer',
        width: '16px',
        height: '16px',
        left: `${state.x + 8}px`,
        top: `${state.y + state.height - 18}px`
      });

      mx.mxEvent.addListener(editImg, 'click', mx.mxUtils.bind(this, (evt: Event) => {
        const version = currentDiagram!.version;
        Streamlit.setComponentValue({
          command: "edit",
          diagram: JSON.stringify(convertMxGraphToDiagramUpdate(graph, version)),
          selected_node: state.cell.id,
          sequence_number: uuidv4()
        });
        mx.mxEvent.consume(evt);
      }));

      graph.container.appendChild(editImg);
      this.images.push(editImg);


      // Create Delete Icon
      const deleteImg: HTMLImageElement = mx.mxUtils.createImage("delete.png");
      deleteImg.setAttribute('title', 'Delete');
      Object.assign(deleteImg.style, {
        position: 'absolute',
        cursor: 'pointer',
        width: '16px',
        height: '16px',
        left: `${state.x + state.width - 16 - 8}px`,
        top: `${state.y + state.height - 18}px`
      });

      // Add event listeners for Delete Icon
      mx.mxEvent.addListener(deleteImg, 'click', mx.mxUtils.bind(this, (evt: MouseEvent) => {
        // eslint-disable-next-line no-restricted-globals
        if (evt.shiftKey || confirm("Delete Node?")) {
          graph.removeCells([this.state.cell], true);

          // remove the output node for the cell too
          const outputNode = graph.getModel().getCell(`output-${this.state.cell.id}`);
          if (outputNode) {
            graph.removeCells([outputNode], true);
          }

          streamlitResponse()
          mx.mxEvent.consume(evt);
          this.destroy()
        }
      }));

      graph.container.appendChild(deleteImg);
      this.images.push(deleteImg);
    } else if (state.cell.isEdge()) {
      const deleteImg: HTMLImageElement = mx.mxUtils.createImage("delete.png");
      deleteImg.setAttribute('title', 'Delete');
      
      const labelBounds = state.text.bounds;
      const labelRightX = labelBounds.x + labelBounds.width;
      const labelBottomY = labelBounds.y + labelBounds.height;

      const imageWidth = 16;
      const imageHeight = 16;

      const offsetX = 0 // -imageWidth; // Positions the image within the label bounds
      const offsetY = 0 // -imageHeight;

      Object.assign(deleteImg.style, {
        position: 'absolute',
        cursor: 'pointer',
        width: `${imageWidth}px`,
        height: `${imageHeight}px`,
        left: `${labelRightX + offsetX}px`,
        top: `${labelBottomY + offsetY}px`,
      });


      // Add event listeners for Delete Icon
      mx.mxEvent.addListener(deleteImg, 'click', mx.mxUtils.bind(this, (evt: MouseEvent) => {
        // eslint-disable-next-line no-restricted-globals
        if (evt.shiftKey || confirm("Delete Edge?")) {
          graph.removeCells([state.cell], false);
        }
        mx.mxEvent.consume(evt);
        this.destroy();
      }));

      // Add event listener for Mouse Over (Hover In)
      mx.mxEvent.addListener(deleteImg, 'mouseenter', mx.mxUtils.bind(this, (evt: MouseEvent) => {
        // Example: Change opacity on hover
        deleteImg.style.opacity = '0.7';
        console.log("Beep")

        // Alternatively, you could change the image source to a hover version
        // deleteImg.src = "delete-hover.png";

        // Or add a CSS class for more complex styling
        // deleteImg.classList.add('delete-icon-hover');
      }));

      // Add event listener for Mouse Out (Hover Out)
      mx.mxEvent.addListener(deleteImg, 'mouseleave', mx.mxUtils.bind(this, (evt: MouseEvent) => {
        // Reset opacity when not hovering
        deleteImg.style.opacity = '1.0';
        console.log("UnBeep")

        // If you changed the image source, revert it back
        // deleteImg.src = "delete.png";

        // Or remove the CSS class
        // deleteImg.classList.remove('delete-icon-hover');
      }));




      graph.container.appendChild(deleteImg);
      this.images.push(deleteImg);
    }
  }

  // Method to remove all icons from the DOM
  destroy(): void {
    if (this.images) {
      this.images.forEach(img => {
        if (img.parentNode) {
          img.parentNode.removeChild(img);
        }
      });
    }
    this.images = null;
  }
}

var currentState: mxCellState | undefined = undefined;
var currentIconSet: mxIconSet | undefined = undefined;

function dragEnter(evt: Event, state: mxCellState) {
  if (can_edit && currentIconSet === undefined) {
    currentIconSet = new mxIconSet(state);
  }
}

function dragLeave(evt: Event, state: mxCellState) {
  if (currentIconSet !== undefined) {
    currentIconSet.destroy();
    currentIconSet = undefined;
  }
}

// Override convertValueToString to handle custom types
graph.convertValueToString = function (...args): string {
  const [cell] = args;

  const value: any = cell.value;

  if (isDiagramNode(value)) {
    return `<span style="font-size:12px;"><b>${value.pill}</b><br></span> ${value.label}`;
  }

  // Default label rendering for non-custom types
  return mx.mxGraph.prototype.convertValueToString.apply(this, args);
};


// Returns the editing value for the given cell and event
graph.getEditingValue = function (cell, evt) {
  if (isDiagramNode(cell.value)) {
    return cell.value.label
  }
  return mx.mxGraph.prototype.getEditingValue.apply(this, [cell, evt]);
};

function cleanReachableNodes(
  startingVertex: mxCell
): void {
  const visited: Set<string> = new Set<string>();
  const stack: mxCell[] = [startingVertex];

  while (stack.length > 0) {
    const current: mxCell | undefined = stack.pop();

    if (!current) continue;

    // Skip if already visited
    if (visited.has(current.id)) {
      continue;
    }
    visited.add(current.id);

    // Check if the vertex ID does NOT start with "output-"
    if (!current.id.startsWith("output-")) {
      // Update style based on phase
      const style = node_style + `fillColor=${clean_color()};`;
      current.value.phase = 0;
      graph.setCellStyle(style, [current]);
    }

    // Get all outgoing edges from the current vertex
    const outgoingEdges: mxCell[] = graph.getOutgoingEdges(current);

    for (const edge of outgoingEdges) {
      // Get the target vertex of the edge
      const target: mxCell = edge.target;

      // If target exists and hasn't been visited, add to the stack
      if (target && !visited.has(target.id)) {
        stack.push(target);
      }
    }
  }
}

// collect list of all node pills in graph.
function collectNodePills(): string[] {
  const pills = [];
  const cells = graph.getModel().cells;
  for (var key in cells) {
    if (cells.hasOwnProperty(key)) {
      var cell = cells[key];
      if (isDiagramNode(cell.value)) {
        pills.push(cell.value.pill);
      }
    }
  }
  return pills;
}


// Sets the new value for the given cell and trigger
graph.labelChanged = function (cell, newValue, trigger): mxCell {

  if (isDiagramNode(cell.value)) {
    graph.setCellsEditable(false);
    // getCompletion(
    //   `Summarize the following text with two words.  
    //   Capitalize each word and hyphenate them.  
    //   Do not include any other text in your response.  
    //   Do not use any of the following: ${collectNodePills().join(", ")}.

    //   Phrase: ${newValue}
    //   `).then((completion) => {
        const value = cell.cloneValue();
        value.label = newValue.trim();
        value.pill = generateTwoWordSummary(newValue);
        cell = mx.mxGraph.prototype.labelChanged.apply(this, [cell, value, trigger]);

        // change label of every outgoing edge to match the new label
        graph.getModel().beginUpdate();
        const edges = graph.getModel().getOutgoingEdges(cell);
        edges.forEach(edge => {
          mx.mxGraph.prototype.labelChanged.apply(this, [edge, labelForEdge(cell), trigger]);
        });
        // not ideal, but we must give the graph view up to date with the changes we're making to phases when
        // this goes back to the driver.
        cleanReachableNodes(cell);
        graph.getModel().endUpdate();
        graph.setCellsEditable(true);
      // })
    return cell
  }
  return mx.mxGraph.prototype.labelChanged.apply(this, [cell, newValue, trigger]);
};


// Add mouse listeners to handle icon display on hover
graph.addMouseListener({

  mouseDown: function (sender: any, me: any) {
    // Hide icons on mouse down
    if (currentState) {
      dragLeave(me.getEvent(), currentState);
      currentState = undefined;
    }
  },

  mouseMove: function (sender: any, me: any) {
    if (currentState && (me.getState() === currentState || me.getState() === null)) {
      const tol = 40;
      const tmp = new mx.mxRectangle(
        me.getGraphX() - tol,
        me.getGraphY() - tol,
        2 * tol,
        2 * tol
      );

      if (mx.mxUtils.intersects(tmp, currentState!)) {
        return;
      }
    }

    var tmpState: mxCellState | undefined = graph.view.getState(me.getCell());

    if (tmpState && tmpState.cell.id.startsWith("output")) {
      tmpState = undefined;
    }

    // Ignore everything but vertices
    if (graph.isMouseDown) { //  || (tmpState && !graph.getModel().isVertex(tmpState.cell))) {
      tmpState = undefined;
    }

    if (tmpState !== currentState) {
      if (currentState) {
        dragLeave(me.getEvent(), currentState);
      }

      currentState = tmpState;

      if (currentState) {
        dragEnter(me.getEvent(), currentState);
      }
    }
  },

  mouseUp: function (sender: any, me: any) { /* No action needed on mouse up */ },

});

//////////////////////////////////////


// Tracking variables
let hoverTimer: number | null = null;
let currentlyHoveredCell: mxCell | null = null;

/**
 * Determines if a cell should trigger the hover event.
 * @param cell - The cell to evaluate.
 * @returns True if the cell is a vertex and its name does not start with "output-", else false.
 */
function shouldHandleHover(cell: mxCell): boolean {
  if (graph.getModel().isVertex(cell)) {
    const id = cell.id; // Adjust based on how the name is stored
    return !id.startsWith('output-');
  }
  return false;
}

/**
 * Handles the hover event by calling hoverNode with the appropriate flag.
 * @param isEntering - True if entering hover, false if exiting.
 */
function handleHover(isEntering: boolean): void {
  let node: string | null = null;
  if (isEntering && currentlyHoveredCell) {
    node = currentlyHoveredCell?.id;
    graph.toggleCellStyle("shadow", false, currentlyHoveredCell);
  } else {
    if (currentlyHoveredCell) {
      graph.toggleCellStyle("shadow", true, currentlyHoveredCell);
      // graph.setCellStyle(node_style, [currentlyHoveredCell]);
    }
    const cells = graph.getSelectionCells();
    const selectedIds = cells.map(cell => cell.id);
    node = selectedIds.length === 0 ? null : selectedIds[0];
  }
  console.log("Hovering: ", currentlyHoveredCell, node)
  const diagram_str = JSON.stringify(convertMxGraphToDiagramUpdate(graph, currentDiagram!.version));
  Streamlit.setComponentValue({
    command: "update",
    diagram: diagram_str,
    selected_node: node,
  });
}

// Custom mouse listener
const hoverMouseListener = {
  mouseDown: (_sender: any, _me: any): void => {
    // No action needed on mouse down
  },

  mouseMove: (_sender: any, me: any): void => {
    const x: number = me.getGraphX();
    const y: number = me.getGraphY();
    const cell: mxCell | null = graph.getCellAt(x, y);

    if (cell !== currentlyHoveredCell) {
      // Clear existing hover timer if any
      if (hoverTimer !== null) {
        clearTimeout(hoverTimer);
        hoverTimer = null;
      }

      // If there was a previously hovered cell, handle exit hover
      if (currentlyHoveredCell && shouldHandleHover(currentlyHoveredCell)) {
        handleHover(false); // Exiting hover
        currentlyHoveredCell = null;
      }

      // If the new cell should handle hover, set a timer
      if (cell && shouldHandleHover(cell)) {
        hoverTimer = window.setTimeout(() => {
          currentlyHoveredCell = cell;
          handleHover(true); // Entering hover
          hoverTimer = null;
        }, 250); // 250 milliseconds delay
      }
    }
  },

  mouseUp: (_sender: any, _me: any): void => {
    // No action needed on mouse up
  }
};

// Add the custom mouse listener to the graph
graph.addMouseListener(hoverMouseListener);

// Handle mouse leaving the graph container
graph.container.addEventListener('mouseout', (_event: MouseEvent): void => {
  // Clear any existing hover timer
  if (hoverTimer !== null) {
    clearTimeout(hoverTimer);
    hoverTimer = null;
  }

  // If a cell is currently hovered, handle exit hover
  if (currentlyHoveredCell && shouldHandleHover(currentlyHoveredCell)) {
    handleHover(false); // Exiting hover
    currentlyHoveredCell = null;
  }
});


//////////////////////////////////////


graph.connectionHandler.setCreateTarget(true);
graph.connectionHandler.connectImage = new mx.mxImage('connect-hollow.png', 24, 24);


function findFirstUnusedStepId() {
  var i = 1;
  var cells = graph.getModel().cells;

  while (true) {
    var idToCheck = "Step-" + i;
    var isUsed = false;

    for (var key in cells) {
      if (cells.hasOwnProperty(key)) {
        var cell = cells[key];
        if (cell.id === idToCheck) {
          isUsed = true;
          break;
        }
      }
    }

    if (!isUsed) {
      return idToCheck;
    }

    i++;
  }
}

graph.connectionHandler.isCreateTarget = function (evt) {
  return evt.currentTarget != null && (evt.currentTarget as HTMLElement).getAttribute("id") === 'graph-container';
}



graph.connectionHandler.createTargetVertex = function (evt, source) {
  const point = graph.getPointForEvent(evt, false);
  const vertex = graph.getModel().cloneCell(source);
  vertex.geometry.x = point.x - vertex.geometry.width / 2;
  vertex.geometry.y = point.y - vertex.geometry.height / 2;
  vertex.id = findFirstUnusedStepId();

  vertex.value = {
    id: vertex.id,
    pill: vertex.id,
    label: "...",
    geometry: vertex.geometry,
    phase: 0
  }
  vertex.setStyle(node_style);

  // Add the new vertex to the graph
  const newCell = graph.addCell(vertex);

  // Prompt the user to enter a label
  const userLabel = prompt("Describe this step", "...");

  if (userLabel != null) {
    generatePill(userLabel, newCell);
  }
  // must clear this before the response is generated
  mouseDown = false;
  streamlitResponse();
  return newCell;
}


// TODO: can we replace this with graph.isMouseDown???
var mouseDown = false;

function streamlitResponse(hover_node: string | null = null) {
  console.log("Streamlit Response", can_edit, mouseDown, graph.isMouseDown)
  if (can_edit && !mouseDown) {
    const cells = graph.getSelectionCells();
    const selectedIds = cells.map(cell => cell.id);
    const selected_node = selectedIds.length === 0 ? null : selectedIds[0];

    const selected_node_value = hover_node == null ? selected_node : hover_node;

    if (currentDiagram !== undefined) {
      const original_version = currentDiagram.version;
      console.log("Setting Value: " + selected_node)
      sessionStorage.setItem("selected_node", selected_node == null ? "" : selected_node);
      const diagram_str = JSON.stringify(convertMxGraphToDiagramUpdate(graph, original_version));
      Streamlit.setComponentValue({
        command: "update",
        diagram: diagram_str,
        selected_node: selected_node_value,
      });
      Streamlit.setFrameHeight();
      graph.sizeDidChange();
    } else {
      console.log("currentPage is undefined")
    }
  }
}

var mouseMoved = false;

function addListeners() {
  graph.addMouseListener({
    mouseDown: (sender, evt) => {
      mouseDown = true;
      mouseMoved = false;
    },
    mouseMove: (sender, evt) => {
      // No operation needed on mouse move
      mouseMoved = true;
    },
    mouseUp: (sender, evt) => {
      mouseDown = false;
      const me = evt.getEvent();
      if (me.shiftKey && !mouseMoved) {
        const userLabel = prompt("Describe this step", "...");
        if (userLabel != null) {
          const pt = graph.getPointForEvent(me, false);
          const model: mxGraphModel = graph.getModel();
          const parent = graph.getDefaultParent();
          model.beginUpdate();
          try {
            const id = findFirstUnusedStepId()

            const width = 160;
            const height = 80;

            // Prompt the user to enter a label
            let value = {
              id: id,
              pill: id,
              label: '...',
              geometry: new mx.mxRectangle(pt.x - width / 2, pt.y - height / 2, width, height),
              phase: 0
            }
            const newCell = graph.insertVertex(parent, id, value, pt.x, pt.y, 160, 80, node_style);
            if (userLabel != null) {
              const value2 = newCell.cloneValue();
              value2.label = userLabel.trim();
              // value.pill = completion;
              graph.getModel().setValue(newCell, value2);
            }    

            generatePill(userLabel, newCell);
          } finally {
            model.endUpdate();
          }
        }
        evt.consume();
      }
      streamlitResponse();
    }
  });

  // Add a listener for selection changes
  graph.getSelectionModel().addListener("change", (sender, evt) => {
    streamlitResponse();
  });


  /**
   * Checks whether adding an edge from source to target would create a cycle in the graph.
   *
   * @param graph - The mxGraph instance.
   * @param source - The source cell from which the edge would originate.
   * @param target - The target cell to which the edge would point.
   * @returns True if adding the edge would create a cycle, false otherwise.
   */
  function wouldCreateCycle(graph: any, source: any, target: any): boolean {
    if (!graph || !source || !target) {
      throw new Error("Graph, source, and target must be provided.");
    }

    // Helper function to perform Depth-First Search (DFS)
    const hasPath = (current: any, target: any, visited: Set<any>): boolean => {
      if (current === target) {
        return true;
      }

      visited.add(current);

      // Get outgoing edges from the current cell
      const outgoingEdges: any[] = graph.getOutgoingEdges(current);

      for (const edge of outgoingEdges) {
        const next = graph.getModel().getTerminal(edge, false); // false for source -> target

        if (!visited.has(next)) {
          if (hasPath(next, target, visited)) {
            return true;
          }
        }
      }

      return false;
    };

    // Check if there's a path from target to source
    const visitedNodes = new Set<any>();
    return hasPath(target, source, visitedNodes);
  }


  graph.getEdgeValidationError = function (...args) {
    const [, source, target] = args;
    if (source == null || target == null) {
      return '';
    }
    if (source.id === target.id) {
      return '';
    }
    if (wouldCreateCycle(graph, source, target)) {
      return 'Edges cannot create cycles in the graph.';
    }
    if (graph.getModel().getEdgesBetween(source, target, false).length > 0) {
      return '';
    }
    return mx.mxGraph.prototype.getEdgeValidationError.apply(this, args);
  };

  // Adds a handler for edge creation
  graph.connectionHandler.insertEdge = function (parent, id, value, source, target, style) {
    if (source != null && target != null) {
      const model: mxGraphModel = graph.getModel();
      model.beginUpdate();
      let edge: mxCell | null = null;
      try {
        // find the result_variable of the source node
        cleanReachableNodes(source);
        const src_function_result_var = labelForEdge(source);
        edge = graph.insertEdge(parent, id, src_function_result_var, source, target, style);
      } finally {
        model.endUpdate();
      }
      mouseDown = false;   // clear hear, in advance of the response
      streamlitResponse();
      return edge;
    }
    return null as any;
  };



  // Add a handler for the Delete key to remove selected cells
  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') {
      graph.clearSelection();
      streamlitResponse();
    } else if (event.key == 'Enter') {
      const cell = graph.getSelectionCell();
      if (cell) {
        graph.startEditingAtCell(cell);
        mx.mxEvent.consume(event);
      }
    }
  });
}


function generateTwoWordSummary(phrase: string): string {
  // List of common English stop words to exclude
  const stopWords: Set<string> = new Set([
      'a', 'an', 'the', 'and', 'or', 'but', 'if', 'in', 'on', 'with', 'for',
      'of', 'at', 'by', 'from', 'up', 'about', 'into', 'over', 'after',
      'under', 'above', 'to', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
      'should', 'can', 'could', 'may', 'might', 'must'
  ]);

  // Step 1: Normalize the input
  const cleanedPhrase = phrase.toLowerCase().replace(/[^a-z0-9\s]/g, '');

  // Step 2: Split into words
  const words = cleanedPhrase.split(/\s+/);

  // Step 3: Remove stop words
  const significantWords = words.filter(word => word.length > 0 && !stopWords.has(word));

  // Step 4: Select the first two significant words
  const summaryWords = significantWords.slice(0, 2);

  let pill : string;

  // If not enough significant words, fallback to first two words
  if (summaryWords.length < 2) {
      const fallbackWords = words.filter(word => word.length > 0).slice(0, 2);
      pill = fallbackWords.map(capitalize).join('-');
  } else {
    const titleCased = summaryWords.map(capitalize);
    pill = titleCased.join('-');
  }
  const pills = collectNodePills()
  console.log(pills)
  if (pills.includes(pill)) {
    var pillNumber = 1;
    while (pills.includes(pill + "-" + pillNumber)) {
      pillNumber++;
    }
    return pill + "-" + pillNumber;
  } else {
    return pill;
  }
}


function capitalize(word: string): string {
  if (word.length === 0) return '';
  return word[0].toUpperCase() + word.slice(1);
}


function generatePill(userLabel: string, newCell: mxCell) {
    const value = newCell.cloneValue();
    value.label = userLabel.trim();
    value.pill = generateTwoWordSummary(userLabel);
    graph.getModel().setValue(newCell, value);

}

function updateGraphWithDiagram(diagram: mxDiagram) {
  console.log("Updating graph with diagram")

  if (currentDiagram !== diagram) {
    currentDiagram = diagram;
    updateDiagram(graph, diagram)
    Streamlit.setFrameHeight();
  }
}

/*****/

function clearGraph() {
  var model = graph.getModel();
  model.beginUpdate();
  try {
      model.clear();
  } finally {
      model.endUpdate();
  }
  graph.refresh();
}


/**
* The component's render function. This will be called immediately after
* the component is initially loaded, and then again every time the
* component gets new data from Python.
*/
function onRender(event: Event): void {
  // console.log("Render Event", event)
  // console.log("Current page", currentPage)
  const data = (event as CustomEvent<RenderData>).detail;
  const initial_render = currentDiagram === undefined;

  setEditable(false);

  let diagram = data.args["diagram"];
  let key = data.args["key"];


  if (data.args["clear"]) {
    clearGraph();
    currentDiagram = undefined;
    sessionStorage.setItem("selected_node", "");
    sessionStorage.setItem("key", "");
  }

  let forced = data.args["forced"] || data.args["clear"];

  currentRefreshPhase = data.args["refresh_phase"];
  // console.log(currentRefreshPhase)
  // console.log(data.args["selected_node"])
  console.log(diagram.version, currentDiagram?.version)
  if (initial_render || !data.args['editable'] || forced) {
    console.log("Initial render")
    updateGraphWithDiagram(diagram)
    const stored_key = sessionStorage.getItem("key");
    if (key === stored_key) {
      const selected_node = sessionStorage.getItem("selected_node")
      console.log("Selected Node: ", selected_node)
      if (selected_node !== null) {
        const model = graph.getModel();
        const cellToSelect = model.getCell(selected_node);
        if (cellToSelect !== null) {
          graph.selectionModel.setCells([cellToSelect]);
          streamlitResponse();
        } else {
          graph.clearSelection();
        }
      }
    } else {
      sessionStorage.setItem("key", key)
      sessionStorage.setItem("selected_node", "")
    }
  } else {
    // updateGraphWithDiagram(diagram)
  }

  if (data.args["selected_node"] === "<<<<<") {
    graph.clearSelection();
    sessionStorage.setItem("selected_node", "");
  }

  console.log("Selected Node: ", graph.getSelectionCells().map(cell => cell.id))

  setEditable(data.args["editable"]);

  // We tell Streamlit to update our frameHeight after each render event, in
  // case it has changed. (This isn't strictly necessary for the example
  // because our height stays fixed, but this is a low-cost function, so
  // there's no harm in doing it redundantly.)
  Streamlit.setFrameHeight();

}

console.log("**** Starting mxgraph component ****")

// Attach our `onRender` handler to Streamlit's render event.
Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);

addListeners();

// Tell Streamlit we're ready to start receiving data. We won't get our
// first RENDER_EVENT until we call this function.
Streamlit.setComponentReady();

// Finally, tell Streamlit to update our initial height. We omit the
// `height` parameter here to have it default to our scrollHeight.
Streamlit.setFrameHeight();

