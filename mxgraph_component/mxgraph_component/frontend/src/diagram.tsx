import { get } from 'lodash';
import mx from './mxgraph';
import { mxCell, mxGeometry, mxGraph, mxHierarchicalLayout } from 'mxgraph';
import { cacheImage, getCachedImage } from './cache';

export interface Geometry {
    x: number;
    y: number;
    width: number;
    height: number;
}

// the value for a node is "<b>pill</b>: label"
export interface DiagramNode {
    id: string;
    kind: number;
    pill: string;
    label: string;
    geometry: Geometry;
    output_geometry: Geometry;
    is_locked: boolean;
    force_show_output: boolean;

    // Extended from dfg
    phase: number;
    has_messages: boolean;
    output?: DiagramOutput;
    build_status?: string;
    html? : string;
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

/**
 * Updated DiagramGroup now includes a geometry field for the expanded state.
 */
export interface DiagramGroup {
    id: string;
    label: string;
    geometry: Geometry;        // Geometry when the group is expanded.
    nodes: string[];           // IDs of nodes in the group. Subgroups are not included here.
    is_collapsed: boolean;
    collapsed_geometry?: Geometry; // Geometry when the group is collapsed.
    parent_group?: string;         // ID of the parent group, or undefined if this is top-level.
}

export interface mxDiagram {
    version: number;
    nodes: { [key: string]: DiagramNode };
    edges: { [key: string]: DiagramEdge };
    groups: DiagramGroup[];
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
    constructor(public pill: string, public label: string) { }

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

export const node_style = 'html=1;whiteSpace=wrap;rounded=0;';
export const node_hover_style = 'html=1;shape=rectangle;whiteSpace=wrap;rounded=1;shadow=1;';

const edge_style = 'endArrow=classic;html=1;rounded=0;labelBackgroundColor=white;';
const output_node_style_text = `html=1;shape=rectangle;whiteSpace=wrap;shadow=1;fillColor=#F0F0F0;strokeColor=#990000;align=left;verticalAlign=middle;spacing=5;fontFamily=monospace;overflow=hidden;`;
const output_node_style_image = `html=1;shape=image;shadow=1;imageBackground=#F0F0F0;imageBorder=#990000;`;
const output_edge_style = 'rounded=1;orthogonalLoop=1;dashed=1;strokeWidth=2;strokeColor=#990000;fillColor=#76608a;endArrow=block;endFill=0;edgeStyle=orthogonalEdgeStyle;curved=0;';
// const phase_colors = [
//     '#F4F4F4', // clean
//     '#d899b3', // requirements
//     '#eeb1b8', // algorithm
//     '#fac4b3', // code
//     '#fedebf', // runnable
//     '#fef2d0', // run_checked
//     '#f5fbd5',
//     '#ddf1da',
//     '#c1e6db',
//     '#adcfe4',
//     '#beb8d9'
// ];

const phase_colors = [
    '#dfedf7', // clean
    '#dfedf7', // requirements
    '#dfedf7', // algorithm
    '#dfedf7', // code
    '#dfedf7', // runnable
    '#fef2d0', // run_checked
    '#fef2d0', // checks compiled
    '#ddf1da', // checked
    '#c1e6db',
    '#adcfe4',
    '#beb8d9'
];


export function phase_to_color(phase: number, build_status: string | undefined): string {
    if (build_status) {
        return '#fac4b3';
    }
    if (phase < 0 || phase >= phase_colors.length) {
        return '#FFFFFF';
    } else {
        return phase_colors[phase];
    }
}

export function clean_color() {
    return phase_to_color(0, undefined);
}

function kind_to_shape(kind: number): string {
    switch (kind) {
        case 0:
            return 'shape=rectangle;rounded=1;';
        case 1:
            return 'shape=ellipse;perimeter=ellipsePerimeter;';
        case 2:
            return 'shape=doubleBorder;';
        default:
            return 'shape=hexagon;';
    }
}

export function style_for_node(node: DiagramNode): string {
    let style = node_style + kind_to_shape(node.kind) + `fillColor=${phase_to_color(node.phase, node.build_status)};`;
    if (node.has_messages) {
        style += "shape=label;strokeColor=#DD0000;strokeWidth=4;imageAlign=center;imageVerticalAlign=middle;imageWidth=80;imageHeight=80;image=error.png;";
    }
    return style;
}

function encodeSVG(svg: string): string {
    return encodeURIComponent(svg)
        .replace(/'/g, "%27")
        .replace(/"/g, "%22");
}

export function getClosestPointOnEllipse(width: number, height: number): { x: number, y: number } {
    const cx = width / 2;
    const cy = height / 2;
    // The closest point is given by (cx/√2, cy/√2)
    return { x: cx / 2 / Math.sqrt(2), y: cy / 2 / Math.sqrt(2) };
  }

function update_overlays_for_node(graph: mxGraph, node: DiagramNode, vertex: mxCell): void {
    graph.removeCellOverlays(vertex);
    if (node.build_status != null) {
        const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="200" height="50">
            <text x="10" y="25" font-family="Arial,Helvetica,sans-serif" font-size="20" fill="#333">
                ${node.build_status}...
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
    if (node.is_locked) {
        const iconImage = new mx.mxImage("lock.png", 16, 16);
        const node_width = vertex.geometry.width;
        const node_height = vertex.geometry.height;
        // const inset_factor = Math.min(node_width, node_height) / 15;
        const iconOverlay = new mx.mxCellOverlay(iconImage, "This node is locked");
        iconOverlay.align = mx.mxConstants.ALIGN_LEFT;
        iconOverlay.verticalAlign = mx.mxConstants.ALIGN_TOP;
        // if (node.kind !== 1) {
            iconOverlay.offset = new mx.mxPoint(16,12);
        // } else {
        //     const closest = getClosestPointOnEllipse(node_width, node_height);
        //     iconOverlay.offset = new mx.mxPoint(closest.x, closest.y);
        // }
        graph.addCellOverlay(vertex, iconOverlay);
    }
}

function make_diagram_node(graph: mxGraph, node: DiagramNode): mxCell {
    const style = style_for_node(node);
    const label = node; // stored as an object so that later we can update properties
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
        const cellId = `output-${node.id}`;
        let vertex: mxCell;
        if (output.output_type === 'text') {
            vertex = graph.insertVertex(
                graph.getDefaultParent(),
                cellId,
                { data: output.data, pill: node.pill },
                node.output_geometry.x,
                node.output_geometry.y,
                node.output_geometry.width,
                node.output_geometry.height,
                style
            );
        } else if (output.output_type === 'image') {
            let image = output.data;
            if (image === 'cached') {
                console.log("Using cached image for node", node.id, node.pill);
                image = getCachedImage(node.id)!;
            } else {
                cacheImage(node.id, image);
            }
            vertex = graph.insertVertex(
                graph.getDefaultParent(),
                cellId,
                { data: '', pill: node.pill },
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
        const cell = graph.insertEdge(
            graph.getDefaultParent(),
            `output-edge-${cellId}`,
            '',
            graph.getModel().getCell(node.id),
            vertex,
            output_edge_style
        );
        return vertex;
    } else {
        return undefined;
    }
}

export function toSnakeCase(input: string): string {
    return input
        .toLowerCase()
        .replace(/[\s\W-]+/g, '_')
        .replace(/^_+|_+$/g, '');
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
    graph.getModel().setValue(cell, node);
    const style = style_for_node(node);
    graph.setCellStyle(style, [cell]);
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

function update_output_node(graph: mxGraph, node: DiagramNode): mxCell | undefined {
    const cell = graph.getModel().getCell(`output-${node.id}`);
    if (!cell) return;
    const output = node.output;
    if (output) {
        if (output.output_type === 'text') {
            graph.getModel().setValue(cell, { data: output.data, pill: node.pill });
        } else if (output.output_type === 'image') {
            let image = output.data;
            if (image === 'cached') {
                console.log("Using cached image for node", node.id, node.pill);
                image = getCachedImage(node.id)!;
            } else {
                cacheImage(node.id, image);
            }
            const style = output_node_style_image + `;image=${image}`;
            graph.setCellStyle(style, [cell]);
            graph.getModel().setValue(cell, { data: '', pill: node.pill });
        }
        const geometry = node.output_geometry;
        if (cell.geometry.x !== geometry.x ||
            cell.geometry.y !== geometry.y ||
            cell.geometry.width !== geometry.width ||
            cell.geometry.height !== geometry.height) {
            const newGeometry = new mx.mxGeometry(geometry.x, geometry.y, geometry.width, geometry.height);
            graph.getModel().setGeometry(cell, newGeometry);
        }
    }
    return cell;
}

function process_output_node(graph: mxGraph, node: DiagramNode, show: boolean): void {
    if (node.output) {
        const cell = graph.getModel().getCell(`output-${node.id}`);
        let output_cell: mxCell | undefined;
        if (cell) {
            output_cell = update_output_node(graph, node);
        } else {
            output_cell = make_output_node(graph, node);
        }
        output_cell!.setVisible(show || node.force_show_output);
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
    graph.getModel().setValue(cell, escapeHtml(labelForEdge(cell.source)));
    graph.setCellStyle(edge_style, [cell]);
}


export const group_style = 'shape=swimlane;strokeWidth=3;startSize=0;verticalAlign=bottom;spacingTop=10;margin=10;whiteSpace=wrap;fontSize=14;fontStyle=1;strokeColor=#880088;';
export const group_collapsed_style = 'shape=swimlane;strokeWidth=3;startSize=0;verticalAlign=top;spacingTop=10;margin=10;whiteSpace=wrap;fontSize=14;fontStyle=1;strokeColor=#880088;';

function lighten(col: string, amt: number) {

    var usePound = false;

    if (col[0] == "#") {
        col = col.slice(1);
        usePound = true;
    }

    var num = parseInt(col, 16);

    var r = (num >> 16) * amt;

    if (r > 255) r = 255;
    else if (r < 0) r = 0;

    var b = ((num >> 8) & 0x00FF) * amt;

    if (b > 255) b = 255;
    else if (b < 0) b = 0;

    var g = (num & 0x0000FF) * amt;

    if (g > 255) g = 255;
    else if (g < 0) g = 0;

    return (usePound ? "#" : "") + (g | (b << 8) | (r << 16)).toString(16);

}


export function update_group_style(graph: mxGraph, cell: mxCell) {
    let color;
    if (cell.isCollapsed()) {
        const model = graph.getModel();
        const nodes = model.getChildCells(cell, true, false);
        const phases = nodes.map(x => (x.value as DiagramNode).phase);
        const min_phase = Math.min(...phases);
        color = `swimlaneFillColor=${phase_to_color(min_phase, undefined)};`;
    } else {
        color = ''
    }
    const style = (cell.isCollapsed() ? group_collapsed_style : group_style) + color;
    graph.setCellStyle(style, [cell]);
    console.log("Updated group style", cell.id, style);

}

/**
 * CustomHierarchicalLayout extends mxHierarchicalLayout.
 *
 * It overrides:
 *
 * 1. execute(parent):
 *    To store the container being laid out (currentParent). When laying out the main graph,
 *    currentParent equals graph.getDefaultParent(). For group layouts, currentParent is that group.
 *
 * 2. getEdges(cell):
 *    For any cell that is a group (has children), it iterates through all children—even if hidden
 *    because the group is collapsed—and adds any edge that connects to a cell outside the group.
 *
 * 3. getVisibleTerminal(edge, source):
 *    When laying out the top‑level graph (currentParent is the default parent), if the terminal of an
 *    edge is inside any group (regardless of collapsed state) then the group container is returned.
 *    When laying out inside a group, the actual terminal is returned.
 */
class CustomHierarchicalLayout extends mx.mxHierarchicalLayout {
    currentParent: any;

    constructor(graph: any) {
        super(graph);
    }

    // Override execute() to store the container being laid out.
    execute(parent: any): void {
        this.currentParent = parent;
        this.edgeStyle = 3;
        super.execute(parent);
    }

    // Helper: Returns true if cell is a descendant of parent.
    isCellDescendant(cell: any, parent: any): boolean {
        while (cell != null) {
            if (cell === parent) {
                return true;
            }
            cell = this.graph.getModel().getParent(cell);
        }
        return false;
    }

    // Override getEdges():
    // For any cell that is a group (has children), regardless of collapsed or expanded,
    // iterate over its children and add any edge that connects to an external cell.
    getEdges(cell: any): any[] {
        const edges: any[] = super.getEdges(cell);
        if (this.graph.getModel().getChildCount(cell) > 0) {
            const childCount = this.graph.getModel().getChildCount(cell);
            for (let i = 0; i < childCount; i++) {
                const child = this.graph.getModel().getChildAt(cell, i);
                const childEdges = this.graph.getModel().getEdges(child);
                if (childEdges != null) {
                    for (let j = 0; j < childEdges.length; j++) {
                        const edge = childEdges[j];
                        const source = this.graph.getModel().getTerminal(edge, true);
                        const target = this.graph.getModel().getTerminal(edge, false);
                        // If either terminal is not a descendant of cell, then the edge is external.
                        if (!this.isCellDescendant(source, cell) || !this.isCellDescendant(target, cell)) {
                            if (edges.indexOf(edge) < 0) {
                                edges.push(edge);
                            }
                        }
                    }
                }
            }
        }
        return edges;
    }

    // Override getVisibleTerminal():
    // When laying out the top‑level graph (currentParent is the default parent),
    // if an edge's terminal is inside one or more groups then climb the parent chain
    // until reaching the outermost group (i.e. the group cell that is a direct child of the default parent).
    // When laying out inside a group (currentParent !== default parent), simply return the actual terminal.
    getVisibleTerminal(edge: any, source: boolean): any {
        const cell = this.graph.getModel().getTerminal(edge, source);
        const parent = cell.getParent();
        const default_parent = this.graph.getDefaultParent();


        if (parent === default_parent) {
            return cell;  // outermost cell
        } else {
            if (source) {
                if (parent.isCollapsed()) {
                    return parent;  // pointing out of collapsed group
                } else {
                    return cell;  // pointing out of expanded group
                }
            } else {
                if (parent.isCollapsed()) {
                    return parent;  // pointing into collapsed group
                } else {
                    return cell; // pointing into expanded group
                }
            }
        }
    }
}


export function layoutDiagram(graph: mxGraph) {

    graph.getModel().beginUpdate();
    try {
        const layout = new mx.mxHierarchicalLayout(graph);
        layout.execute(graph.getDefaultParent());
    } finally {
        graph.getModel().endUpdate();
    }
}



    // graph.getModel().beginUpdate();
    // try {
    //     // console.log(graph)

    //     // add an implicit edge from each node to the group of the target if that gropu is not the default parent
    //     const cells = graph.getChildCells(graph.getDefaultParent(), true, true);
    //     for (const cell of cells) {
    //         // find edges to point into some group
    //         const edges = graph.getModel().getEdges(cell);
    //         for (const edge of edges) {
    //             const source = edge.source;
    //             const target = edge.target;
    //             if (source && target) {
    //                 if (target.getParent() !== graph.getDefaultParent()) {
    //                     const newEdge = graph.insertEdge(graph.getDefaultParent(), 'null', 'implicit', source, target.getParent());
    //                 }
    //                 if (source.getParent() !== graph.getDefaultParent()) {
    //                     const newEdge = graph.insertEdge(graph.getDefaultParent(), 'null', 'implicit', source.getParent(), target);
    //                 }
    //             }
    //         }
    //     }

    //     const parent = graph.getDefaultParent();
    //     const layout = new CustomHierarchicalLayout(graph);
    //     layout.execute(parent);

    //     const childCount = graph.getModel().getChildCount(parent);
    //     for (let i = 0; i < childCount; i++) {
    //         const cell = graph.getModel().getChildAt(parent, i);
    //         if (graph.getModel().getChildCount(cell) > 0 && !graph.isCellCollapsed(cell)) {
    //             const layoutGroup = new CustomHierarchicalLayout(graph);
    //             layoutGroup.execute(cell);
    //             graph.updateGroupBounds([cell], 20);
    //         }
    //     }

    //     // remove all those implicit edges
    //     const edges = graph.getChildEdges(graph.getDefaultParent());
    //     for (const edge of edges) {
    //         if (edge.value === 'implicit') {
    //             graph.getModel().remove(edge);
    //         }
    //     }
    //     // console.log(graph)
    // } finally {
    //     graph.getModel().endUpdate();
    // }
// }


// /**
//  * Recursively applies a hierarchical layout to the given cell if it has children.
//  * This ensures that nodes inside a group are laid out only within that group's boundaries.
//  *
//  * @param {mxCell} cell The cell (group) to check for children and run the layout.
//  */
// function runNestedLayout(graph: mxGraph, cell: mxCell) {
//     if (!graph.isCellCollapsed(cell) && graph.getModel().getChildCount(cell) > 0) {
//         var groupLayout = makeLayout(graph);
//         groupLayout.execute(cell);
//     }
// }

// /**
//  * Updates the bounds of all group cells so that each group (if expanded)
//  * is resized to enclose all its children with a specified border.
//  */
// function updateAllGroupBounds(graph: mxGraph) {
//     function updateGroupRecursively(cell : mxCell) {
//         // Process nested groups first (bottom-up)
//         var children = graph.getModel().getChildCells(cell, true, false);
//         for (var j = 0; j < children.length; j++) {
//             if (graph.getModel().getChildCount(children[j]) > 0) {
//                 updateGroupRecursively(children[j]);
//             }
//         }

//         if (cell.collapsed && cell.manualCollapsedSize) {
//             graph.getModel().setGeometry(cell, cell.manualCollapsedSize.clone());
//         } else {
//             graph.updateGroupBounds([cell], 20, false);
//         }
//     }

//     // Process all top-level group cells.
//     var cells = graph.getModel().getChildCells(graph.getDefaultParent(), true, false);
//     for (var i = 0; i < cells.length; i++) {
//         if (graph.getModel().getChildCount(cells[i]) > 0) {
//             updateGroupRecursively(cells[i]);
//         }
//     }
// }

/**
 * Adjusts the geometry of collapsed groups so that their height is
 * reduced to just the header (as defined by the style's STARTSIZE).
 * This function is called recursively on nested groups.
 *
 * @param {mxCell} cell The group cell to update.
 */
function updateCollapsedGroups(graph: mxGraph, cell: mxCell) {
    var model = graph.getModel();
    if (model.isVertex(cell) && model.getChildCount(cell) > 0) {
        if (graph.isCellCollapsed(cell)) {
            if (cell.manualCollapsedSize) {
                graph.getModel().setGeometry(cell, cell.manualCollapsedSize.clone());
            } else {
                var style = graph.getCellStyle(cell);
                var startSize = parseInt(style[mx.mxConstants.STYLE_STARTSIZE]) || 30;
                var geo = model.getGeometry(cell);
                if (geo != null) {
                    geo = geo.clone();
                    geo.height = startSize;
                    model.setGeometry(cell, geo);
                }
            }
        } else {
            // Process any nested groups.
            var children = model.getChildCells(cell, true, false);
            for (var i = 0; i < children.length; i++) {
                updateCollapsedGroups(graph, children[i]);
            }
        }
    }
}

// function runLayout(graph: mxGraph) {
//     const layout = makeLayout(graph);
//     layout.execute(graph.getDefaultParent());

//     updateAllGroupBounds(graph);

//     var cells = graph.getModel().getChildCells(graph.getDefaultParent(), true, false);

//     for (var i = 0; i < cells.length; i++) {
//         runNestedLayout(graph, cells[i]);
//     }
//     updateAllGroupBounds(graph);

//     // // Now adjust collapsed groups to only display the header area.
//     // var topGroups = graph.getModel().getChildCells(graph.getDefaultParent(), true, false);
//     // for (var i = 0; i < topGroups.length; i++) {
//     //     updateCollapsedGroups(graph,topGroups[i]);
//     // }
// }


// export function layoutDiagram(graph: mxGraph) {
//     const parent = graph.getDefaultParent();
//     graph.getModel().beginUpdate();
//     try {
//         runLayout(graph)
//         resetNodeTranslation(graph);
//     } catch (error) {
//         console.error("Error applying layout:", error);
//     } finally {
//         const morph = new mx.mxMorphing(graph);
//         morph.addListener('done', function () {
//             graph.getModel().endUpdate();
//         });
//         morph.startAnimation();
//     }
//     graph.refresh();
// }

// function makeLayout(graph: mxGraph) {
//     const layout = new mx.mxHierarchicalLayout(graph, "north");
//     layout.interRankCellSpacing = 35;
//     layout.disableEdgeStyle = true;
//     layout.edgeStyle = 3;
//     return layout;
// }

export function resetNodeTranslation(graph: mxGraph) {
    const parent = graph.getDefaultParent();
    const cells = graph.getChildCells(parent, true, true);
    if (cells.length > 0) {
        let minX = Infinity, minY = Infinity;
        for (const cell of cells) {
            const bounds = cell.geometry;
            if (bounds) {
                minX = Math.min(minX, bounds.x);
                minY = Math.min(minY, bounds.y);
            }
        }
        graph.view.setTranslate(-minX + 20, -minY + 20);
    }
}

/**
 * Updates the mxGraph to match the given mxDiagram.
 *
 * In addition to nodes and edges, we now process groups. The new group handling uses
 * group.geometry when expanded and group.collapsed_geometry when collapsed.
 */
export function updateDiagram(graph: mxGraph, diagram: mxDiagram): void {
    const model = graph.getModel();
    model.beginUpdate();
    try {
        const parent = graph.getDefaultParent();

        // === Handle Nodes ===
        const existingNodeIds = new Set<string>();
        const existingEdgeIds = new Set<string>();

        for (const nodeId in diagram.nodes) {
            const node = diagram.nodes[nodeId];
            existingNodeIds.add(nodeId);
            let cell = model.getCell(node.id);
            if (cell) {
                update_node(graph, node);
            } else {
                cell = make_diagram_node(graph, node);
            }
        }

        // Remove nodes that are no longer present.
        const cells = model.getDescendants(parent);
        if (cells !== null) {
            for (let i = 0; i < cells.length; i++) {
                const cell = cells[i];
                const cellId = cell.id;
                // Exclude output cells and group cells (group ids start with "group-")
                if (cell.isVertex() && !cellId.startsWith('output-') && !cellId.startsWith('group-')) {
                    if (!existingNodeIds.has(cellId)) {
                        graph.removeCells([cell], true);
                        const outputCell = model.getCell(`output-${cellId}`);
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
            let edgeCell = model.getCell(edge.id);
            if (edgeCell) {
                update_edge(graph, edge);
            } else {
                try {
                    edgeCell = make_edge(graph, edge);
                } catch (error) {
                    console.error(`Failed to create edge ${edge.id}:`, error);
                }
            }
        }
        if (cells !== null) {
            for (let i = 0; i < cells.length; i++) {
                const cell = cells[i];
                if (cell.isEdge()) {
                    const edgeId = cell.id;
                    if (!existingEdgeIds.has(edgeId) && !edgeId.startsWith('output-edge-')) {
                        graph.removeCells([cell], false);
                    }
                }
            }
        }

        for (const nodeId in diagram.nodes) {
            const node = diagram.nodes[nodeId];
            let hasChild = false;
            for (const edgeId in diagram.edges) {
                const edge = diagram.edges[edgeId];
                if (edge.src === node.id) {
                    hasChild = true;
                    break;
                }
            }
            process_output_node(graph, node, !hasChild);
        }

        // Layout if any node has zero geometry.
        let layoutNeeded = false;
        for (const nodeId in diagram.nodes) {
            const node = diagram.nodes[nodeId];
            console.log("Checking node", node.id, node.geometry);
            if (node.geometry.x === 0 && node.geometry.y === 0 &&
                node.geometry.width === 0 && node.geometry.height === 0) {
                node.geometry = { x: 0, y: 0, width: 120, height: 60 };
                node.output_geometry = { x: 0, y: 0, width: 120, height: 120 };
                update_node(graph, node);
                layoutNeeded = true;
            }
        }

        // === Handle Groups ===
        // Remove any group cells no longer present.
        const existingGroupIds = new Set(diagram.groups.map(g => g.id));
        const allCells = model.getDescendants(parent);
        for (const cell of allCells) {
            if (cell.vertex && cell.id.startsWith("group-") && !existingGroupIds.has(cell.id)) {
                console.log("Removing group cell", cell.id);
                graph.removeCells([cell], true);
            }
        }
        // Update or add groups.
        for (const group of diagram.groups) {
            console.log("Processing group", group.id);
            console.log(model.getDescendants(parent));
            const descendents = model.getDescendants(parent);
            let groupCell = descendents.find(node => node.id === group.id);
            console.log("Group cell", group.id, groupCell);
            if (groupCell) {
                console.log("Updating group cell", group.id);
                model.setValue(groupCell, group.label);
            } else {
                console.log("Creating group cell", group.id);
                const members = group.nodes.map(nodeId => model.getCell(nodeId));
                groupCell = graph.groupCells(null, 20, members);
                graph.addCell(groupCell, parent);
                model.setGeometry(groupCell, new mx.mxGeometry(0, 0, 200, 150));
                groupCell.setId(group.id);
                groupCell.setStyle('group');
                groupCell.value = group.label;
                groupCell.setConnectable(false);
            }

            update_group_style(graph, groupCell);

            // Reparent member nodes into the group.
            for (const nodeId of group.nodes) {
                const nodeCell = model.getCell(nodeId);
                if (nodeCell && nodeCell.parent !== groupCell) {
                    graph.model.add(groupCell, nodeCell);
                }
            }
            // If this group is nested, reparent it.
            if (group.parent_group) {
                const parentGroupCell = model.getCell(group.parent_group);
                if (parentGroupCell && groupCell.parent !== parentGroupCell) {
                    graph.model.add(parentGroupCell, groupCell);
                }
            }
        }
        for (const group of diagram.groups) {
            let groupCell = model.getCell(group.id);
            console.log("Running nested layout for group", group.id, groupCell);
            graph.foldCells(group.is_collapsed, false, [groupCell]);
        }

        if (layoutNeeded) {
            layoutDiagram(graph);
        }


    } finally {
        model.endUpdate();
    }
}


/**
 * The update format now also carries groups.
 */
export interface DiagramNodeUpdate {
    id: string;
    pill: string;
    label: string;
    geometry: Geometry;
    output_geometry?: Geometry;
    is_locked: boolean;
    force_show_output: boolean;
}

export interface DiagramEdgeUpdate {
    id: string;
    src: string;
    dst: string;
}

// The update interface now includes groups.
export interface mxDiagramUpdate {
    version: number;
    nodes: { [key: string]: DiagramNodeUpdate };
    edges: { [key: string]: DiagramEdgeUpdate };
    groups: DiagramGroup[];
}

/**
 * Converts an mxGraph instance back into the mxDiagramUpdate format.
 * In addition to nodes and edges, we now also collect all group cells.
 */
export function convertMxGraphToDiagramUpdate(graph: mxGraph, original_version: number): mxDiagramUpdate {
    const diagramUpdate: mxDiagramUpdate = {
        version: original_version,
        nodes: {},
        edges: {},
        groups: []
    };

    const model = graph.getModel();
    const parent = graph.getDefaultParent();
    const allCells = model.getDescendants(parent);

    for (const cell of allCells) {
        if (model.isEdge(cell)) {
            if (!cell.id.startsWith("output-edge-")) {
                diagramUpdate.edges[cell.id] = { id: cell.id, src: cell.source.id, dst: cell.target.id };
            }
        } else if (cell.vertex) {
            if (cell.id.startsWith("output-")) {
                continue;
            } else if (cell.id.startsWith("group-")) {
                // Process group cell.
                const groupId = cell.id;
                const groupLabel = (typeof cell.value === 'string')
                    ? cell.value
                    : (cell.value && cell.value.label ? cell.value.label : '');
                // Note: use getChildCells with recursive=false to get immediate children.
                const childCells = model.getChildCells(cell, true, false);
                const nodeIds: string[] = [];
                for (const child of childCells) {
                    if (child.vertex && !child.id.startsWith("output-") && !child.id.startsWith("group-")) {
                        nodeIds.push(child.id);
                    }
                }
                const isCollapsed = !!cell.collapsed;
                const collapsedGeometry = isCollapsed ? {
                    x: cell.geometry.x,
                    y: cell.geometry.y,
                    width: cell.geometry.width,
                    height: cell.geometry.height
                } : undefined;
                // Determine the expanded geometry.
                let expandedGeometry: Geometry;
                if (!cell.collapsed && cell.geometry) {
                    expandedGeometry = {
                        x: cell.geometry.x,
                        y: cell.geometry.y,
                        width: cell.geometry.width,
                        height: cell.geometry.height
                    };
                } else if ((cell as any).expandedGeometry) {
                    const eg = (cell as any).expandedGeometry;
                    expandedGeometry = { x: eg.x, y: eg.y, width: eg.width, height: eg.height };
                } else {
                    expandedGeometry = { x: 0, y: 0, width: 200, height: 150 };
                }
                let parent_group: string | undefined = undefined;
                if (cell.parent && cell.parent !== parent && cell.parent.id && cell.parent.id.startsWith("group-")) {
                    parent_group = cell.parent.id;
                }
                const diagramGroup: DiagramGroup = {
                    id: groupId,
                    label: groupLabel,
                    geometry: expandedGeometry,
                    nodes: nodeIds,
                    is_collapsed: isCollapsed,
                    collapsed_geometry: collapsedGeometry,
                    parent_group: parent_group
                };
                console.log("Adding group", diagramGroup);
                diagramUpdate.groups.push(diagramGroup);
            } else {
                // Process node cell.
                const nodeId = cell.id;
                const rawLabel = cell.value as DiagramNode;
                const geometry: Geometry = {
                    x: cell.geometry.x,
                    y: cell.geometry.y,
                    width: cell.geometry.width,
                    height: cell.geometry.height
                };
                let output_geometry: Geometry | undefined;
                const outputCell = model.getCell(`output-${nodeId}`);
                if (outputCell) {
                    output_geometry = {
                        x: outputCell.geometry.x,
                        y: outputCell.geometry.y,
                        width: outputCell.geometry.width,
                        height: outputCell.geometry.height
                    };
                }
                const is_locked = rawLabel.is_locked;
                const force_show_output = rawLabel.force_show_output;
                diagramUpdate.nodes[nodeId] = {
                    id: nodeId,
                    pill: rawLabel.pill,
                    label: rawLabel.label,
                    geometry: geometry,
                    output_geometry: output_geometry,
                    is_locked: is_locked,
                    force_show_output: force_show_output
                };
            }
        }
    }
    return diagramUpdate;
}
