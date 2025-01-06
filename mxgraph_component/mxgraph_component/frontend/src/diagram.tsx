import { Streamlit } from 'streamlit-component-lib';
import mx from './mxgraph';
import { mxCell, mxGraph, mxGeometry, mxHierarchicalLayout, mxMorphing, mxEvent } from 'mxgraph';
import { v4 as uuidv4 } from "uuid";

export interface Geometry {
    x: number;
    y: number;
    width: number;
    height: number;
}

// the value for a node is "<b>pill</b>: label"
export interface DiagramNode {
    id: string;
    pill: string;
    label: string;
    geometry: Geometry;
    output_geometry: Geometry;

    // Extended from dfg
    phase: number;
    has_question: boolean;
    has_messages: boolean;
    output?: DiagramOutput;
    build_status?: string;

    html: string;
    
}


export interface DiagramOutput {
    output_type: string;
    data: string;
}

// the value for an edge is the pill.
export interface DiagramEdge {
    id: string;
    pill: string;
    src: string;
    dst: string;
}

export interface mxDiagram {
    version: number;
    nodes: { [key: string]: DiagramNode };
    edges: { [key: string]: DiagramEdge };
}

function isGeometry(obj: any): obj is Geometry {
    return (
        typeof obj === "object" &&
        obj !== null &&
        typeof obj.x === "number" &&
        typeof obj.y === "number" &&
        typeof obj.width === "number" &&
        typeof obj.height === "number"
    );
}

function isDiagramOutput(obj: any): obj is DiagramOutput {
    return (
        typeof obj === "object" &&
        obj !== null &&
        typeof obj.outputId === "string" &&
        typeof obj.outputType === "string"
    );
}

export function isDiagramNode(obj: any): obj is DiagramNode {
    return (
        typeof obj === "object" &&
        obj !== null &&
        typeof obj.id === "string" &&
        typeof obj.pill === "string" &&
        typeof obj.label === "string" 
    );
}


export class NodeValue {
    constructor(public pill: string, public label: string) {
    }

    html(): string {
        return `<b>${this.pill}</b><br>${this.label}`;
    }

}

function escapeHtml(text: string): string {
    const map: { [key: string]: string } = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;',
    };
    return text.replace(/[&<>"']/g, function (m) { return map[m]; });
}


export const node_style = 'html=1;shape=rectangle;whiteSpace=wrap;rounded=1;';
export const node_hover_style = 'html=1;shape=rectangle;whiteSpace=wrap;rounded=1;shadow=1';
const edge_style = 'endArrow=classic;html=1;rounded=0;labelBackgroundColor=white';
const output_node_style_text = `html=1;shape=rectangle;whiteSpace=wrap;shadow=1;fillColor=#E8E8E8;strokeColor=#990000;align=left;verticalAlign=middle;spacing=5;fontFamily=monospace;`;
const output_node_style_image = `html=1;shape=image;shadow=1;imageBackground=#E8E8E8;imageBorder=#990000;`;
const output_edge_style = 'rounded=1;orthogonalLoop=1;dashed=1;strokeWidth=2;strokeColor=#990000;fillColor=#76608a;endArrow=block;endFill=0;edgeStyle=orthogonalEdgeStyle;curved=0;';
const phase_colors = [
    '#F4F4F4', // clean
    '#d899b3', // requirements
    '#eeb1b8', // algorithm
    '#fac4b3', // code
    '#fedebf', // runnable
    '#fef2d0', // run_checked
    '#f5fbd5', 
    '#ddf1da', 
    '#c1e6db',
    '#adcfe4', 
    '#beb8d9'];

function phase_to_color(phase: number): string {
    console.log('phase', phase);
    if (phase < 0 || phase >= phase_colors.length) {
        return '#FFFFFF';
    } else {
        return phase_colors[phase];
    }
}

export function clean_color() {
    return phase_to_color(0);
}

function style_for_node(node: DiagramNode): string {
    var style = node_style + `fillColor=${phase_to_color(node.phase)};`;
    if (node.has_messages) {
        style = style + "shape=label;strokeColor=#DD0000;strokeWidth=4;imageAlign=center;imageVerticalAlign=middle;imageWidth=80;imageHeight=80;image=error.png;";
    }

    return style
}

function encodeSVG(svg: string): string {
    return encodeURIComponent(svg)
        .replace(/'/g, "%27")
        .replace(/"/g, "%22");
}

function update_overlays_for_node(graph: mxGraph, node: DiagramNode, vertex: mxCell): void {
    graph.removeCellOverlays(vertex);    
    if (node.build_status != null) {    
        console.log('build_status', node.build_status);
        const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="200" height="50">
            <text x="10" y="25" font-family="Arial,Helvetica,sans-serif" font-size="20" fill="#333">${node.build_status}...
                          <animate attributeName="opacity" values="0.8;0.2;0.8" dur="2s" begin="0s" repeatCount="indefinite" />
                </text>
          </svg>`;
    
        const encodedSVG = encodeSVG(svg);
        const dataUri = `data:image/svg+xml;charset=UTF-8,${encodedSVG}`;
        const overlayImage = new mx.mxImage(dataUri, 200 * 0.5, 50 * 0.5);
        const overlay = new mx.mxCellOverlay(overlayImage, node.build_status);
        overlay.align = mx.mxConstants.ALIGN_LEFT;
        overlay.verticalAlign = mx.mxConstants.ALIGN_TOP;
        overlay.offset = new mx.mxPoint(50, -5);
        graph.addCellOverlay(vertex, overlay);
    }    

}


function make_diagram_node(graph: mxGraph, node: DiagramNode): mxCell {
    const style = style_for_node(node);
    const label = node; // `<b>${escapeHtml(node.pill)}</b>: ${escapeHtml(node.label)}`;
    const vertex = graph.insertVertex(
        graph.getDefaultParent(),
        node.id,
        label,
        node.geometry.x,
        node.geometry.y,
        node.geometry.width,
        node.geometry.height,
        style
    );
    update_overlays_for_node(graph, node, vertex);
    vertex.setConnectable(true);
    return vertex;
}

function make_output_node(graph: mxGraph, node: DiagramNode): mxCell | undefined {
    const output = node.output;

    if (output !== undefined) {
        const style = output.output_type === 'text' ? output_node_style_text : output_node_style_image;
        const label = output.output_type === 'text' ? output.data : '';
        const image = output.output_type === 'image' ? output.data : '';

        const cellId = `output-${node.id}`;
        let vertex: mxCell;

        if (output.output_type === 'text') {
            vertex = graph.insertVertex(
                graph.getDefaultParent(),
                cellId,
                label,
                node.output_geometry.x,
                node.output_geometry.y,
                node.output_geometry.width,
                node.output_geometry.height,
                style
            );
        } else if (output.output_type === 'image') {
            vertex = graph.insertVertex(
                graph.getDefaultParent(),
                cellId,
                '',
                node.output_geometry.x,
                node.output_geometry.y,
                node.output_geometry.width,
                node.output_geometry.height,
                `${style};image=${image}`
            );
        } else {
            throw new Error(`Unsupported output type: ${output.output_type}`);
        }

        vertex.setConnectable(false);
        // Optionally, connect the output to the node
        const cell = graph.insertEdge(
            vertex.parent,
            `output-edge-${cellId}`,
            '',
            graph.getModel().getCell(node.id),
            vertex,
            output_edge_style
        );
        console.log(cell);

        return vertex;
    } else {
        return undefined;
    }
}


export function toSnakeCase(input: string): string {
    return input
        .toLowerCase()
        .replace(/[\s\W-]+/g, '_') // Replace spaces, punctuation, and hyphens with an underscore
        .replace(/^_+|_+$/g, ''); // Trim leading/trailing underscores if any
}

export function labelForEdge(src_node: mxCell): string {
    return toSnakeCase(src_node ? escapeHtml((src_node.value as DiagramNode).pill) : "");
}


function make_edge(graph: mxGraph, edge: DiagramEdge): mxCell {
    const srcCell = graph.getModel().getCell(edge.src);
    const dstCell = graph.getModel().getCell(edge.dst);

    if (!srcCell || !dstCell) {
        throw new Error(`Source or destination cell not found for edge ${edge.id}`);
    }

    const edgeCell = graph.insertEdge(
        graph.getDefaultParent(),
        edge.id,
        labelForEdge(srcCell),
        srcCell,
        dstCell,
        edge_style
    );

    return edgeCell;
}


function update_node(graph: mxGraph, node: DiagramNode): void {
    const cell = graph.getModel().getCell(node.id);
    if (!cell) return;

    // Update label
    graph.getModel().setValue(cell, node);

    console.log('update node', node);

    // Update style based on phase
    const style = style_for_node(node);
    graph.setCellStyle(style, [cell]);

    // Update geometry if changed
    const geometry = node.geometry;
    if (cell.geometry.x !== geometry.x ||
        cell.geometry.y !== geometry.y ||
        cell.geometry.width !== geometry.width ||
        cell.geometry.height !== geometry.height) {
        const newGeometry = new mx.mxGeometry(geometry.x, geometry.y, geometry.width, geometry.height);
        graph.getModel().setGeometry(cell, newGeometry);
    }

    update_overlays_for_node(graph, node, cell);
}


function update_output_node(graph: mxGraph, node: DiagramNode): void {
    const cell = graph.getModel().getCell(`output-${node.id}`);
    if (!cell) return;
    const output = node.output;
    if (output) {
        if (output.output_type === 'text') {
            graph.getModel().setValue(cell, output.data);
        } else if (output.output_type === 'image') {
            const style = output_node_style_image + `;image=${output.data}`;
            graph.setCellStyle(style, [cell]);
        }

        // Update geometry if changed
        const geometry = node.output_geometry;
        if (cell.geometry.x !== geometry.x ||
            cell.geometry.y !== geometry.y ||
            cell.geometry.width !== geometry.width ||
            cell.geometry.height !== geometry.height) {
            const newGeometry = new mx.mxGeometry(geometry.x, geometry.y, geometry.width, geometry.height);
            graph.getModel().setGeometry(cell, newGeometry);
        }
    }
}

function process_output_node(graph: mxGraph, node: DiagramNode, show: boolean): void {
    if (node.output && show) {
        const cell = graph.getModel().getCell(`output-${node.id}`);
        if (cell) {
            update_output_node(graph, node);
        } else {
            make_output_node(graph, node);
        }
        // alt: graph.toggleCells(show, [cell], true);
    } else {
        const cell = graph.getModel().getCell(`output-${node.id}`);
        if (cell) {
            graph.removeCells([cell], true);
        }
    }
}

function update_edge(graph: mxGraph, edge: DiagramEdge): void {
    const cell = graph.getModel().getCell(edge.id);
    if (!cell) return;

    // Update label
    graph.getModel().setValue(cell, escapeHtml(labelForEdge(cell.source)));

    // Update style if needed
    graph.setCellStyle(edge_style, [cell]);
}

function layoutDiagram(graph: mxGraph) {
    // Get the default parent for inserting cells
    const parent = graph.getDefaultParent();
  
    graph.getModel().beginUpdate();
    try {
        console.log('layoutDiagram');
        // Use mxHierarchicalLayout for a nice hierarchical arrangement
        const layout = new mx.mxHierarchicalLayout(graph, "west");
        layout.execute(parent);
        

        // Calculate the left-most and top-most node positions
        const cells = graph.getChildCells(parent, true, true);
        if (cells.length > 0) {
            let minX = Infinity, minY = Infinity;
            for (const cell of cells) {
                const bounds = cell.geometry
                if (bounds) {
                    minX = Math.min(minX, bounds.x);
                    minY = Math.min(minY, bounds.y);
                }
                console.log(minX, minY);
            }
            // Set the view translation to inset the nodes by 20 pixels
            graph.view.setTranslate(-minX + 20, -minY + 20);
        }

    } catch (error) {
        console.error("Error applying layout:", error);
    } finally {
        var morph = new mx.mxMorphing(graph);
        morph.addListener('done', function()
        {
            graph.getModel().endUpdate();
        });
        
        morph.startAnimation();
// graph.getModel().endUpdate();
    }
  
    graph.refresh();
}
  

export function updateDiagram(graph: mxGraph, diagram: mxDiagram): void {
    const model = graph.getModel();
    
    model.beginUpdate();
    console.log('updateDiagram', diagram);
    try {
        const parent = graph.getDefaultParent();

        // === Handle Nodes ===
        const existingNodeIds = new Set<string>();
        const existingEdgeIds = new Set<string>();

        for (const nodeId in diagram.nodes) {
            const node = diagram.nodes[nodeId];
            existingNodeIds.add(nodeId);
            const cellId = node.id;
            let cell = model.getCell(cellId);

            console.log(node.pill)

            if (cell) {
                // Update existing node
                console.log('update node', node);
                update_node(graph, node);
            } else {
                // Add new node
                console.log('add node', node);
                cell = make_diagram_node(graph, node);
            }
        }

        // Remove nodes that are no longer present
        const cells = model.getChildren(parent);
        if (cells !== null) {
            for (let i = 0; i < cells.length; i++) {
                const cell = cells[i];
                const cellId = cell.id;
                if (cell.isVertex() && !cellId.startsWith('output-')) {
                    const nodeId = cellId;
                    if (!existingNodeIds.has(nodeId)) {
                        console.log('remove node', nodeId);
                        graph.removeCells([cell], true);
                        // remove the ocrresponding output node
                        const outputCell = model.getCell(`output-${nodeId}`);
                        if (outputCell) {
                            graph.removeCells([outputCell], true);
                        }
                    }
                }
            }
        }

        // === Handle Edges ===
        for (const edgeId in diagram.edges) {
            const edge = diagram.edges[edgeId];
            existingEdgeIds.add(edgeId);
            const cellId = edge.id;
            let edgeCell = model.getCell(cellId);

            if (edgeCell) {
                // Update existing edge
                update_edge(graph, edge);
            } else {
                // Add new edge
                try {
                    edgeCell = make_edge(graph, edge);
                } catch (error) {
                    console.error(`Failed to create edge ${edge.id}:`, error);
                }
            }
        }

        // Remove edges that are no longer present
        if (cells !== null) {
            for (let i = 0; i < cells.length; i++) {
                const cell = cells[i];
                const cellId = cell.id;
                if (cell.isEdge()) {
                    const edgeId = cellId;
                    if (!existingEdgeIds.has(edgeId) && !cellId.startsWith('output-edge-')) {
                        graph.removeCells([cell], false);
                    }
                }
            }
        }

        for (const nodeId in diagram.nodes) {
            const node = diagram.nodes[nodeId];
            var hasChild = false;
            for (const edgeId in diagram.edges) {
                const edge = diagram.edges[edgeId];
                if (edge.src === node.id) {
                    hasChild = true;
                    break;
                }
            }

            process_output_node(graph, node, !hasChild);
        }

        // if any nodes have geometry 0,0,0,0, then resize them to the default size and call layout_diagram
        var layout = false;
        for (const nodeId in diagram.nodes) {
            const node = diagram.nodes[nodeId];
            if (node.geometry.x === 0 && node.geometry.y === 0 && node.geometry.width === 0 && node.geometry.height === 0) {
                console.log(node.id, 'has 0,0,0,0 geometry');
                node.geometry = { x: 0, y: 0, width: 120, height: 60 };
                node.output_geometry = { x: 0, y: 0, width: 120, height: 120 };
                update_node(graph, node);
                layout = true;
            }
        }
        if (layout) {
            layoutDiagram(graph);
        }

    } finally {
        model.endUpdate();
    }
}


// the value for a node is "<b>pill</b>: label"
export interface DiagramNodeUpdate {
    id: string;
    pill: string;
    label: string;
    geometry: Geometry;
    output_geometry?: Geometry;
}

// the value for an edge is the pill.
export interface DiagramEdgeUpdate {
    id: string;
    src: string;
    dst: string;
}

export interface mxDiagramUpdate {
    version: number
    nodes: { [key: string]: DiagramNodeUpdate };
    edges: { [key: string]: DiagramEdgeUpdate };
}


/**
 * Converts an mxGraph instance back into the mxDiagramUpdate format.
 * @param graph The mxGraph instance to convert.
 * @returns The reconstructed mxDiagramUpdate object.
 */
export function convertMxGraphToDiagramUpdate(graph: mxGraph, original_version: number): mxDiagramUpdate {
    const diagramUpdate: mxDiagramUpdate = {
        nodes: {},
        edges: {},
        version: original_version
    };

    const model = graph.getModel();
    const parent = graph.getDefaultParent();
    const cells = model.getChildren(parent);

    // Iterate through all cells in the graph
    for (const cellId in cells) {
        const cell: mxCell = cells[cellId];

        if (model.isEdge(cell)) {
            // === Handle Edges ===
            const edgeId = cell.id;
            // According to DiagramEdgeUpdate, only 'id' is needed
            if (!cell.id.startsWith("output-edge-")) {
                diagramUpdate.edges[edgeId] = { id: edgeId, src: cell.source.id, dst: cell.target.id };
            }
        } else if (cell.vertex) {
            // === Handle Nodes ===
            if (!cell.id.startsWith("output-")) {
                const nodeId = cell.id; // e.g., "node1"
                const rawLabel = cell.value as DiagramNode
                const label = rawLabel.label;
                const pill = rawLabel.pill;
                const geometry: Geometry = cell.geometry;
                let output_geometry: Geometry | undefined;
                if (model.getCell(`output-${nodeId}`)) {
                    const outputCell = model.getCell(`output-${nodeId}`);
                    output_geometry = outputCell.geometry;
                } else {
                    output_geometry = undefined;
                }
                diagramUpdate.nodes[nodeId] = { id: nodeId, pill: pill, label: label, geometry: geometry, output_geometry: output_geometry };
            }
        }
        // Non-vertex, non-edge cells are ignored as per interface requirements
    }

    return diagramUpdate;
}
