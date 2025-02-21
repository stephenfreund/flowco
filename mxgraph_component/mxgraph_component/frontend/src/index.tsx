import { Streamlit, RenderData } from "streamlit-component-lib";
import mx from './mxgraph';
import { mxGraphModel, mxCellState, mxCell, mxConstants, mxHierarchicalLayout } from 'mxgraph';
import { v4 as uuidv4 } from "uuid";

import { mxDiagram, updateDiagram, convertMxGraphToDiagramUpdate, node_style, labelForEdge, clean_color, isDiagramNode, layoutDiagram, DiagramNode, phase_to_color, update_group_style } from "./diagram";
import { clearImageCache } from "./cache";

var currentDiagram: mxDiagram | undefined = undefined;
var currentRefreshPhase = 0;

// Create a container for the graph
const graphContainer = document.querySelector("#graph-container") as HTMLDivElement;
export const graph = new mx.mxGraph(graphContainer);

// Define zoom factors and limits
const ZOOM_IN_FACTOR = 1.4;  // 10% zoom in
const ZOOM_OUT_FACTOR = 0.6; // 10% zoom out
const MIN_ZOOM = 0.25;        // 25%
const MAX_ZOOM = 3.0;        // 300%


/**
 * Zoom In Function
 */
function zoomIn() {
  let newScale = graph.view.scale * ZOOM_IN_FACTOR;
  if (newScale > MAX_ZOOM) newScale = MAX_ZOOM;
  graph.view.scale = newScale;
  graph.refresh();
}

/**
 * Zoom Out Function
 */
function zoomOut() {
  let newScale = graph.view.scale * ZOOM_OUT_FACTOR;
  if (newScale < MIN_ZOOM) newScale = MIN_ZOOM;
  graph.view.scale = newScale;
  sessionStorage.setItem("scale", JSON.stringify(graph.view.scale));
  graph.refresh();
}

/**
 * Reset Zoom Function
 */
function resetZoom() {
  graph.view.scale = 1.0; // Reset to 100%
  graph.view.translate = new mx.mxPoint(40, 60); // Reset pan
  sessionStorage.setItem("scale", JSON.stringify(graph.view.scale));
  graph.refresh();
}

function getZoomScale() {
  return graph.view.scale;
}

function setZoomScale(scale: number) {
  graph.view.scale = scale;
  sessionStorage.setItem("scale", JSON.stringify(graph.view.scale));
  graph.refresh();
}

function zoom(cmd: string) {
  if (cmd == "in") {
    zoomIn();
  } else if (cmd == "out") {
    zoomOut();
  } else if (cmd == "reset") {
    resetZoom();
  } else {
    // console.log("Unknown zoom command: ", cmd);
  }
}

let zoomedInContainer = document.getElementById("customBox")!;

// Function to display zoomed-in content (with image handling)
function showZoomedInContent(cell: mxCell) {
  // zoomedInContainer.innerHTML = cell.value;  // Clear any previous content
  const style = cell.style;

  // Check if the style contains a background image
  const imageMatch = style && style.match('image=data:image/png,\(.*\)');

  if (imageMatch && imageMatch[1]) {
    const title = `${cell.value.pill} Output`
    const imageUrl = imageMatch[1];
    // console.log("Image URL", imageUrl)

    // Create an <img> element and set the src to the extracted image URL
    const imgElement = document.createElement('img');
    imgElement.src = `data:image/png;base64,${imageUrl}`;
    imgElement.style.maxWidth = '100%'; // Optional: set a maximum size for the image
    zoomedInContainer.innerHTML = `<h4>${title}</h4><br/>${imgElement.outerHTML}`;
  } else {
    // If no image is found, display the cell value as text
    const title = `${cell.value.pill} Output`
    zoomedInContainer.innerHTML = `<h4>${title}</h4><br/>${cell.value.data}`;
  }
  // Show the container and position it near the mouse
  zoomedInContainer.style.display = "block";
}

// Function to display zoomed-in content (with image handling)
function showZoomedInNodeContent(cell: mxCell) {
  const node = cell.value
  const html = node.html 
  
  if (html) {
    zoomedInContainer.innerHTML = html;
    // Show the container and position it near the mouse
    zoomedInContainer.style.display = "block";
  }
}

zoomedInContainer.style.position = 'absolute';
zoomedInContainer.style.alignSelf = 'center';
zoomedInContainer.style.left = '10px;' 
zoomedInContainer.style.margin = '10px';
zoomedInContainer.style.top = '60px';  // Align with the top of the cell
zoomedInContainer.style.width = "95%";
zoomedInContainer.style.fontSize = "12px";
zoomedInContainer.style.boxShadow = "0 0 10px rgba(0, 0, 0, 0.5)";
zoomedInContainer.style.maxWidth = "100%";


function hideZoomedInContent() {
  zoomedInContainer.style.display = 'none';
}

zoomedInContainer.addEventListener("mouseup", (event) => {
  event.stopPropagation();
  hideZoomedInContent();
});

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

// Get the default edge style from the graph's stylesheet
var edgeStyle = graph.getStylesheet().getDefaultEdgeStyle();

// Set the edge style to use the elbow connector
// Remove the connector style so that edges are drawn as straight lines
delete edgeStyle[mx.mxConstants.STYLE_EDGE];


function cellKind(node: mxCell) {
  if (node.isEdge()) {
    return "edge";
  } else if (node.id.startsWith("output-")) {
    return "output";
  } else if (node.id.startsWith("group-")) {
    return "group";
  } else {
    return "node";
  }
}

// Enables rubberband selection
// new mx.mxRubberband(graph);


// Enable panning
// graph.setPanning(true);
graph.panningHandler.useLeftButtonForPanning = true;

// Listen for mouse move events to update the cursor
graph.container.addEventListener('mousemove', function (evt) {
  var offset = mx.mxUtils.getOffset(graph.container);
  var x = mx.mxEvent.getClientX(evt) - offset.x;
  var y = mx.mxEvent.getClientY(evt) - offset.y;
  var state = graph.view.getState(graph.getCellAt(x, y));
  var cell = state ? state.cell : null;
  var isShift = evt.shiftKey;

  // hideZoomedInContent();

  if (cell == null) {
    if (isShift) {
      graph.container.style.cursor = 'default';
    } else {
      graph.container.style.cursor = 'grab';
    }
  } else {
    graph.container.style.cursor = 'default';
  }
});



function setEditable(editable: boolean) {
  can_edit = editable;
  graph.setEnabled(can_edit);
}

graphContainer.addEventListener('contextmenu', (event) => {
  event.preventDefault();
});


// Ensure the graph container is focusable
graphContainer.setAttribute('tabindex', '0');



// Class representing a set of icons displayed on vertex hover
class mxIconSet {
  private images: HTMLImageElement[] | null;

  constructor(private state: mxCellState) {
    console.log("BEEP")
    this.images = [];
    const graph = state.view.graph;

    if (cellKind(state.cell) === "output" || cellKind(state.cell) === "group") {
      return;
    } else if (state.cell.isVertex()) {
      // tset if the current cell and all predecessors have phase 1 or better.
      const upToDate = (cell: mxCell): boolean => {
        if (cell.value.phase < currentRefreshPhase) {
          return false;
        }
        const incomingEdges = graph.getIncomingEdges(cell, graph.getDefaultParent());
        for (const edge of incomingEdges) {
          const source = edge.source;
          if (!upToDate(source)) {
            return false;
          }
        }
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
        if (can_edit && (evt.shiftKey || confirm("Delete Node?"))) {
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


      // if state.cell has any outgoing edges that do not have empty labels, show output.
      const children = graph.getModel().getOutgoingEdges(state.cell);
      const showOutput = children.some(child => !child.value.startsWith("output"));
      if (showOutput) {
        // Create For show
        const showOutputImg: HTMLImageElement = state.cell.value.force_show_output ? mx.mxUtils.createImage("visible_filled.png") : mx.mxUtils.createImage("visible.png")
        showOutputImg.setAttribute('title', 'Show Output');
        Object.assign(showOutputImg.style, {
          position: 'absolute',
          cursor: 'pointer',
          width: '16px',
          height: '16px',
          left: `${state.x + state.width - 16 - 8}px`,
          top: `${state.y + 2}px`
        });

        // Add event listeners for Delete Icon
        mx.mxEvent.addListener(showOutputImg, 'click', mx.mxUtils.bind(this, (evt: MouseEvent) => {
          const node_value = this.state.cell.value
          node_value.force_show_output = !node_value.force_show_output;

          const output_node = graph.getModel().getCell(`output-${this.state.cell.id}`);
          if (output_node) {
            graph.toggleCells(node_value.force_show_output, [output_node], true);
          }

          streamlitResponse();
          mx.mxEvent.consume(evt);
          this.destroy();
          graph.refresh();
        }));
        graph.container.appendChild(showOutputImg);
        this.images.push(showOutputImg);
      }



    } else if (state.cell.isEdge()) {

      const deleteImg: HTMLImageElement = mx.mxUtils.createImage("delete.png");
      deleteImg.setAttribute('title', 'Delete');

      if (state.text != null) {
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
          if (can_edit && (evt.shiftKey || confirm("Delete Edge?"))) {
            graph.removeCells([state.cell], false);
          }
          mx.mxEvent.consume(evt);
          this.destroy();
        }));

        mx.mxEvent.addListener(deleteImg, 'mouseenter', mx.mxUtils.bind(this, (evt: MouseEvent) => {
          deleteImg.style.opacity = '0.7';
        }));

        mx.mxEvent.addListener(deleteImg, 'mouseleave', mx.mxUtils.bind(this, (evt: MouseEvent) => {
          deleteImg.style.opacity = '1.0';
        }));

        graph.container.appendChild(deleteImg);
        this.images.push(deleteImg);
      }
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
    if (graph.getModel().getChildCount(state.cell) === 0) {
      currentIconSet = new mxIconSet(state);
    }
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
    return `<span style="font-size:14px;"><b>${value.pill}</b><br></span> ${value.label}`;
  } else if (typeof value.pill === "string") {
    // output node
    return value.data;
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
    if (cellKind(current) !== "output") {
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
    const value = cell.cloneValue();
    value.label = newValue.trim();
    // value.pill = generateTwoWordSummary(newValue);
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
    streamlitResponse();
    return cell
  } else {
    streamlitResponse();
    return mx.mxGraph.prototype.labelChanged.apply(this, [cell, newValue, trigger]);
  }
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

    if (tmpState && cellKind(tmpState.cell) === "output") {
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

function shouldHandleHover(cell: mxCell): boolean {
  let kind = cellKind(cell);
  return (kind === "output" || kind === "node");
}

/**
 * Handles the hover event by calling hoverNode with the appropriate flag.
 * @param isEntering - True if entering hover, false if exiting.
 */
function handleHover(isEntering: boolean): void {
  console.log("Handling hover", isEntering, currentlyHoveredCell)
  if (!mouseDown) {
    if (isEntering && currentlyHoveredCell) {
      if (cellKind(currentlyHoveredCell) === "output") {
        showZoomedInContent(currentlyHoveredCell);
      } 
      else if (cellKind(currentlyHoveredCell) === "node") {
        showZoomedInNodeContent(currentlyHoveredCell);
      }
    } else {
      if (currentlyHoveredCell) {
        hideZoomedInContent();
        //  else if (cellKind(currentlyHoveredCell) === "node") {
        //   graph.toggleCellStyle("shadow", true, currentlyHoveredCell);
        // }
        // const cells = graph.getSelectionCells();
        // const selectedIds = cells.map(cell => cell.id);
        // node = selectedIds.length === 0 ? null : selectedIds[0];
      }

      // console.log("Hovering: ", currentlyHoveredCell, node)

      // if (!(currentlyHoveredCell && cellKind(currentlyHoveredCell) === "output")) {
      //   const diagram_str = JSON.stringify(convertMxGraphToDiagramUpdate(graph, currentDiagram!.version));
      //   Streamlit.setComponentValue({
      //     command: "update",
      //     diagram: diagram_str,
      //     selected_node: node,
      //   });
      // }
    }
  }
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
        // console.log("Exiting hover", currentlyHoveredCell)    
        handleHover(false); // Exiting hover
        currentlyHoveredCell = null;
      }

      // If the new cell should handle hover, set a timer
      if (cell && shouldHandleHover(cell)) {
        hoverTimer = window.setTimeout(() => {
          currentlyHoveredCell = cell;
          handleHover(true); // Entering hover
          hoverTimer = null;
        }, 500); // 250 milliseconds delay
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
    // console.log("Exiting hover due to mouse out", currentlyHoveredCell)
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
    phase: 0,
    is_locked: false,
    force_show_output: true
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

function streamlitResponse() {
  console.log("Streamlit Response Borp", can_edit, mouseDown, graph.isMouseDown)
  if (can_edit && !mouseDown) {
    const cells = graph.getSelectionCells();
    const selectedIds = cells.map(cell => cell.id);
    const selected_node = selectedIds.length === 0 ? null : selectedIds[0];

    if (currentDiagram !== undefined) {
      const original_version = currentDiagram.version;
      // console.log("Setting Value: " + selected_node)
      sessionStorage.setItem("selected_node", selected_node == null ? "" : selected_node);

      const translation = graph.view.translate;
      sessionStorage.setItem("translation", JSON.stringify(translation));

      const scale = getZoomScale();
      sessionStorage.setItem("scale", JSON.stringify(scale));

      const diagram_str = JSON.stringify(convertMxGraphToDiagramUpdate(graph, original_version));
      Streamlit.setComponentValue({
        command: "update",
        diagram: diagram_str,
        selected_node: selected_node,
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
      if (can_edit && (me.shiftKey && !mouseMoved)) {
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
              phase: 0,
              is_locked: false,
              force_show_output: true
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
    console.log("Selection Changed", graph.getSelectionCells().map(cell => cell.id))
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
    } else if (event.key === 'Enter') {
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

  let pill: string;

  // If not enough significant words, fallback to first two words
  if (summaryWords.length < 2) {
    const fallbackWords = words.filter(word => word.length > 0).slice(0, 2);
    pill = fallbackWords.map(capitalize).join('-');
  } else {
    const titleCased = summaryWords.map(capitalize);
    pill = titleCased.join('-');
  }
  const pills = collectNodePills()
  // console.log(pills)
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




graph.foldingEnabled = true; // Enable collapsible groups

// Only cells with children are considered foldable
graph.isCellFoldable = function (cell, collapse) {
  return this.model.getChildCount(cell) > 0;
};

// ************************************
// Preserve Collapsed Size When Resizing
// ************************************
// When a collapsed group is resized by the user, save the new geometry
// on a custom property (manualCollapsedSize) so that future layout calls remember it.
graph.addListener(mx.mxEvent.RESIZE_CELLS, function (sender, evt) {
  var cells = evt.getProperty('cells');
  for (var i = 0; i < cells.length; i++) {
    var cell = cells[i];
    if (cell.collapsed) {
      var geo = graph.getModel().getGeometry(cell);
      if (geo != null) {
        cell.manualCollapsedSize = geo.clone();
      }
    }
  }
});


// ************************************
// Grouping and Ungrouping Functionality
// ************************************


// /**
//  * Groups the currently selected cells into a new group.
//  * The new group uses the custom 'group' style (swimlane) so that it has an editable header.
//  */
// function groupCells() {
//     var cells = graph.getSelectionCells();
//     if (cells && cells.length > 0) {
//         graph.getModel().beginUpdate();
//         try {
//             // Group the selected cells (they become children of the new group cell)
//             var group = graph.groupCells(null, 0, cells);
//             group.setId("group-" + group.id);
//             if (group != null) {
//                 // Apply the custom group style (swimlane).
//                 group.setStyle('group');
//                 // If no label is set, assign a default one.
//                 if (!group.value) {
//                     group.value = 'Group';
//                 }
//                 // Select the new group cell.
//                 graph.setSelectionCell(group);
//             }
//         } finally {
//             graph.getModel().endUpdate();
//             layoutDiagram(graph);
//         }
//     }
// }

/**
 * Groups the currently selected cells according to these rules:
 *
 * 1. If all selected cells are regular nodes, then create a new group.
 * 2. If one group cell is selected along with one or more regular nodes (that are not already grouped),
 *    then add those nodes to the selected group.
 * 3. Otherwise, alert the user that the selection is invalid.
 */
function groupCells() {
  var cells = graph.getSelectionCells();
  if (!cells || cells.length === 0) {
    return;
  }

  // Helper function to determine if a cell is a group.
  // In this example, a cell is considered a group if its style is 'group'.
  function isGroup(cell: mxCell) {
    return cell.getStyle && cell.getStyle() === 'group';
  }

  // Separate the selection into group cells and regular cells.
  var groupCellsArr = [];
  var regularCellsArr = [];

  for (var i = 0; i < cells.length; i++) {
    var cell = cells[i];
    if (isGroup(cell)) {
      groupCellsArr.push(cell);
    } else {
      regularCellsArr.push(cell);
    }
  }

  // Get the default parent used for ungrouped nodes.
  var defaultParent = graph.getDefaultParent();

  // CASE 1: All selected cells are regular nodes.
  if (groupCellsArr.length === 0 && regularCellsArr.length > 0) {
    graph.getModel().beginUpdate();
    try {
      // Group all the selected regular nodes into a new group.
      var newGroup = graph.groupCells(null, 0, cells);
      if (newGroup != null) {
        // Set a new id and the custom style.
        newGroup.setId("group-" + uuidv4());
        update_group_style(graph, newGroup);
        graph.getModel().setGeometry(newGroup, new mx.mxGeometry(0, 0, 200, 150));
        // If no label is set, assign a default one.
        if (!newGroup.value) {
          newGroup.value = 'Group';
        }
        // Select the new group cell.
        graph.setSelectionCell(newGroup);
      }
    } finally {
      graph.getModel().endUpdate();
      layoutDiagram(graph);
    }
  }
  // CASE 2: One group cell and one or more regular nodes.
  else if (groupCellsArr.length === 1 && regularCellsArr.length > 0) {
    var targetGroup = groupCellsArr[0];
    // Verify that each selected regular node is not already in a group.
    for (var j = 0; j < regularCellsArr.length; j++) {
      var regCell = regularCellsArr[j];
      // A regular node is considered ungrouped if its parent is the default parent.
      if (regCell.parent !== defaultParent) {
        alert("You can only group nodes that are not already part of a group.");
        return;
      }
    }
    // All checks passed; add the regular nodes to the target group.
    graph.getModel().beginUpdate();
    try {
      for (var k = 0; k < regularCellsArr.length; k++) {
        var cellToAdd = regularCellsArr[k];
        // This call moves the cell into the target group.
        graph.getModel().add(targetGroup, cellToAdd, targetGroup.getChildCount());
      }
      // Optionally, update the selection to include both the target group and the newly added nodes.
      graph.setSelectionCells([targetGroup].concat(regularCellsArr));
    } finally {
      graph.getModel().endUpdate();
      layoutDiagram(graph);
    }
  }
  // CASE 3: Any other combination of selections.
  else {
    alert("You cannot group two existing groups.");
  }
}


/**
 * Ungroups each selected group cell.
 */
function ungroupCells() {
  var cells = graph.getSelectionCells();
  if (cells && cells.length > 0) {
    graph.getModel().beginUpdate();
    try {
      for (var i = 0; i < cells.length; i++) {
        if (graph.getModel().getChildCount(cells[i]) > 0) {
          graph.ungroupCells([cells[i]]);
        }
      }
    } finally {
      graph.getModel().endUpdate();
      layoutDiagram(graph);
    }
  }
}

// // ************************************
// // Button Event Listeners
// // ************************************
document.getElementById('groupBtn')!.addEventListener('click', groupCells);
document.getElementById('ungroupBtn')!.addEventListener('click', ungroupCells);


// Add a listener for selection changes
graph.getSelectionModel().addListener(mx.mxEvent.CHANGE, function(sender, evt) {
  // Get the currently selected cells
  var cells = graph.getSelectionCells();
  var btn = document.getElementById('groupBtn')!;
  var ungroupBtn = document.getElementById('ungroupBtn')!;

  // Initially hide the button
  btn.style.display = 'none';
  ungroupBtn.style.display = 'none';

  if (cells.length > 1) {
    var countWithChildren = 0;
    for (var i = 0; i < cells.length; i++) {
      if (graph.getModel().getChildCount(cells[i]) > 0) {
        countWithChildren++;
      }
    }
    if (countWithChildren <= 1) {
      btn.innerText = 'Group';
      btn.style.display = ''; // Show the button
    }
  } else if (cells.length === 1 && graph.getModel().getChildCount(cells[0]) > 0) {
    ungroupBtn.style.display = ''; // Show the button
  }
});

// Re-run layout (and update group bounds) when groups are collapsed/expanded.
graph.addListener(mx.mxEvent.FOLD_CELLS, function (sender, evt) {
  var cells = evt.getProperty('cells');
  var collapsed = evt.getProperty('collapsed');
  for (var i = 0; i < cells.length; i++) {
    var cell = cells[i];
    if (graph.isSwimlane(cell)) {
      update_group_style(graph, cell);
    }
  }

  layoutDiagram(graph);
});

// Assume 'graph' is your mxGraph instance
graph.isCellMovable = function (cell) {
  // If the cell is not in the default parent, then it belongs to a group
  if (cell.parent !== this.getDefaultParent()) {
    return false;
  }
  // Otherwise, use the default behavior
  return mx.mxGraph.prototype.isCellMovable.apply(this, arguments as unknown as [mxCell]);
};




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

  // console.log("currentDiagram === undefined", initial_render)
  // console.log("forced", data.args["forced"])
  // console.log("clear", data.args["clear"])
  // console.log("!editable", !data.args["editable"])
  // console.log("Force", initial_render || !data.args['editable'] || data.args["forced"] || data.args["clear"])


  setEditable(false);

  let diagram = data.args["diagram"];
  let key = data.args["key"];


  if (data.args["clear"]) {
    clearGraph();
    currentDiagram = undefined;
    sessionStorage.setItem("selected_node", "");
    sessionStorage.setItem("key", "");
    // clearImageCache();
  }


  let forced = data.args["forced"] || data.args["clear"];

  currentRefreshPhase = data.args["refresh_phase"];
  // console.log(currentRefreshPhase)
  // console.log(data.args["selected_node"])
  // console.log(diagram.version, currentDiagram?.version)
  if (initial_render || !data.args['editable'] || forced) {
    graph.view.setTranslate(40, 60);

    updateGraphWithDiagram(diagram)
    const stored_key = sessionStorage.getItem("key");
    if (key === stored_key) {
      const selected_node = sessionStorage.getItem("selected_node")
      // console.log("Selected Node: ", selected_node)
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
      const scale = JSON.parse(sessionStorage.getItem("scale") || "1.0");
      setZoomScale(scale);
      const translation = JSON.parse(sessionStorage.getItem("translation") || "{ x: 40, y: 60 }");
      graph.view.setTranslate(translation.x, translation.y);
    } else {
      sessionStorage.setItem("key", key)
      sessionStorage.setItem("selected_node", "")
    }
  } else {
    // updateGraphWithDiagram(diagram)
  }

  // console.log("Borp", data.args['zoom'])
  if (data.args['zoom'] != null) {
    // console.log("Zooming", data.args['zoom'], graph.view.translate, graph.view.scale)
    zoom(data.args['zoom']);
  }


  const translation = graph.view.translate;
  sessionStorage.setItem("translation", JSON.stringify(translation));
  sessionStorage.setItem("scale", JSON.stringify(getZoomScale()));

  if (data.args["selected_node"] === "<<<<<") {
    graph.clearSelection();
    sessionStorage.setItem("selected_node", "");
  }

  // console.log("Selected Node: ", graph.getSelectionCells().map(cell => cell.id))

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
