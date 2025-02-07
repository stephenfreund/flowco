import { get } from 'lodash';
import mx from './mxgraph';
import { mxCell, mxGeometry, mxGraph } from 'mxgraph';
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

export const node_style = 'html=1;shape=rectangle;whiteSpace=wrap;rounded=1;';
export const node_hover_style = 'html=1;shape=rectangle;whiteSpace=wrap;rounded=1;shadow=1;';
// Use a swimlane shape to represent a group.
export const group_style = 'html=1;shape=swimlane;startSize=20;fillColor=#f0f0f0;strokeColor=#888888;rounded=1;';

const edge_style = 'endArrow=classic;html=1;rounded=0;labelBackgroundColor=white;';
const output_node_style_text = `html=1;shape=rectangle;whiteSpace=wrap;shadow=1;fillColor=#E8E8E8;strokeColor=#990000;align=left;verticalAlign=middle;spacing=5;fontFamily=monospace;overflow=hidden;`;
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
    '#beb8d9'
];

function phase_to_color(phase: number): string {
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
    let style = node_style + `fillColor=${phase_to_color(node.phase)};`;
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
        const inset_factor = Math.min(node_width, node_height) / 15;
        const iconOverlay = new mx.mxCellOverlay(iconImage, "This node is locked");
        iconOverlay.align = mx.mxConstants.ALIGN_RIGHT;
        iconOverlay.verticalAlign = mx.mxConstants.ALIGN_TOP;
        iconOverlay.offset = new mx.mxPoint(-4 - inset_factor, 6 + inset_factor);
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
                output.data,
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
        const cell = graph.insertEdge(
            vertex.parent,
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
            graph.getModel().setValue(cell, output.data);
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


/**
 * Recursively applies a hierarchical layout to the given cell if it has children.
 * This ensures that nodes inside a group are laid out only within that group's boundaries.
 *
 * @param {mxCell} cell The cell (group) to check for children and run the layout.
 */
function runNestedLayout(graph: mxGraph, cell: mxCell) {
    if (!graph.isCellCollapsed(cell) && graph.getModel().getChildCount(cell) > 0) {
        var groupLayout = makeLayout(graph);
        groupLayout.execute(cell);
    }
}

/**
 * Updates the bounds of all group cells so that each group (if expanded)
 * is resized to enclose all its children with a specified border.
 */
function updateAllGroupBounds(graph: mxGraph) {
    function updateGroupRecursively(cell : mxCell) {
        // Process nested groups first (bottom-up)
        var children = graph.getModel().getChildCells(cell, true, false);
        for (var j = 0; j < children.length; j++) {
            if (graph.getModel().getChildCount(children[j]) > 0) {
                updateGroupRecursively(children[j]);
            }
        }

        if (cell.collapsed && cell.manualCollapsedSize) {
            graph.getModel().setGeometry(cell, cell.manualCollapsedSize.clone());
        } else {
            graph.updateGroupBounds([cell], 20, false);
        }
    }

    // Process all top-level group cells.
    var cells = graph.getModel().getChildCells(graph.getDefaultParent(), true, false);
    for (var i = 0; i < cells.length; i++) {
        if (graph.getModel().getChildCount(cells[i]) > 0) {
            updateGroupRecursively(cells[i]);
        }
    }
}

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

function runLayout(graph: mxGraph) {
    const layout = makeLayout(graph);
    layout.execute(graph.getDefaultParent());

    updateAllGroupBounds(graph);

    var cells = graph.getModel().getChildCells(graph.getDefaultParent(), true, false);

    for (var i = 0; i < cells.length; i++) {
        runNestedLayout(graph, cells[i]);
    }
    updateAllGroupBounds(graph);

    // // Now adjust collapsed groups to only display the header area.
    // var topGroups = graph.getModel().getChildCells(graph.getDefaultParent(), true, false);
    // for (var i = 0; i < topGroups.length; i++) {
    //     updateCollapsedGroups(graph,topGroups[i]);
    // }
}


export function layoutDiagram(graph: mxGraph) {
    const parent = graph.getDefaultParent();
    graph.getModel().beginUpdate();
    try {
        runLayout(graph)
        resetNodeTranslation(graph);
    } catch (error) {
        console.error("Error applying layout:", error);
    } finally {
        const morph = new mx.mxMorphing(graph);
        morph.addListener('done', function () {
            graph.getModel().endUpdate();
        });
        morph.startAnimation();
    }
    graph.refresh();
}

function makeLayout(graph: mxGraph) {
    const layout = new mx.mxHierarchicalLayout(graph, "north");
    layout.interRankCellSpacing = 35;
    layout.disableEdgeStyle = true;
    layout.edgeStyle = 3;
    return layout;
}

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
            // let geom: mxGeometry;
            // if (group.is_collapsed && group.collapsed_geometry) {
            //     geom = new mx.mxGeometry(group.collapsed_geometry.x, group.collapsed_geometry.y, group.collapsed_geometry.width, group.collapsed_geometry.height);
            // } else {
            //     // Use the expanded geometry.
            //     if (group.geometry) {
            //         geom = new mx.mxGeometry(group.geometry.x, group.geometry.y, group.geometry.width, group.geometry.height);
            //     } else {
            //         // geom should bound the included nodes
            //         const groupCells = group.nodes.map(nodeId => model.getCell(nodeId));
            //         let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
            //         for (const cell of groupCells) {
            //             if (cell) {
            //                 const bounds = cell.geometry;
            //                 if (bounds) {
            //                     minX = Math.min(minX, bounds.x);
            //                     minY = Math.min(minY, bounds.y);
            //                     maxX = Math.max(maxX, bounds.x + bounds.width);
            //                     maxY = Math.max(maxY, bounds.y + bounds.height);
            //                 }
            //             }
            //         }
            //         geom = new mx.mxGeometry(minX, minY, maxX - minX, maxY - minY);
            //     }
            // }
            if (groupCell) {
                console.log("Updating group cell", group.id);
                model.setValue(groupCell, group.label);
            } else {
                console.log("Creating group cell", group.id);
                const members = group.nodes.map(nodeId => model.getCell(nodeId));
                groupCell = graph.groupCells(null, 20, members); 
                graph.addCell(groupCell, parent);
                model.setGeometry(groupCell, new mx.mxGeometry());
                groupCell.setId(group.id);
                groupCell.setStyle('group');
                groupCell.value = group.label;
                groupCell.setConnectable(false);
            }
            // groupCell.collapsed = group.is_collapsed;
            // groupCell.manualCollapsedSize = group.collapsed_geometry ? new mx.mxRectangle(group.collapsed_geometry.x, group.collapsed_geometry.y, group.collapsed_geometry.width, group.collapsed_geometry.height) : undefined;

            // if (group.is_collapsed && group.collapsed_geometry) {
            //     model.setGeometry(groupCell, new mx.mxGeometry(group.collapsed_geometry.x, group.collapsed_geometry.y, group.collapsed_geometry.width, group.collapsed_geometry.height));
            // } else if (group.geometry) {
            //     model.setGeometry(groupCell, new mx.mxGeometry(group.geometry.x, group.geometry.y, group.geometry.width, group.geometry.height));
            // }
            // // Store the expanded geometry as a custom property on the cell.
            // (groupCell as any).expandedGeometry = group.geometry;
            
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

        // if (layoutNeeded) {
        //     layoutDiagram(graph);
        // }


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
