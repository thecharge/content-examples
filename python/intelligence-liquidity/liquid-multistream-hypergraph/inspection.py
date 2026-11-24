"""Export neural memberships and parameter aliases without flattening their meaning."""

import hashlib
import json


def inspect(graph):
    vertices, relations = {}, []
    parameter_owners, layer_owners = {}, {}
    for board, core in sorted(graph.cores.items()):
        vertices[board] = dict(kind="board")
        for expert in core.ids:
            identity = board + "/" + expert
            vertices[identity] = dict(kind="expert", board=board)
            relations.append(dict(kind="contains", members=[board, identity]))
        for layer, cell in zip(core.layer_ids, core.cells):
            local = board + "/" + layer
            canonical = layer_owners.setdefault(id(cell), local)
            vertices[local] = dict(
                kind="liquid_layer", canonical=canonical, board=board
            )
            relations.append(dict(kind="contains", members=[board, local]))
            if local != canonical:
                relations.append(dict(kind="shared_layer", members=[canonical, local]))
            for expert in core.ids:
                for unit in cell.units:
                    identity = board + "/" + expert + "/" + layer + "/u" + str(unit)
                    vertices[identity] = dict(
                        kind="liquid_neuron",
                        board=board,
                        parameter_vertex=canonical + "/" + expert + "/u" + str(unit),
                    )
                    relations.append(
                        dict(
                            kind="contains",
                            members=[board + "/" + expert, local, identity],
                        )
                    )
        for name, node in core.delta.nodes():
            identity = board + "/attention/" + name
            vertices[identity] = dict(
                kind="delta_attention", heads=node.identities, board=board
            )
            relations.append(dict(kind="contains", members=[board, identity]))
    for key, edge in graph.edges.items():
        relations.append(
            dict(kind="subexpert", members=[key, edge["child"]], fields=edge["group"])
        )
    for identity, edge in graph.hyperedges.items():
        members = []
        for vertex in edge["vertices"]:
            member = edge["board"] + "/" + vertex["expert"]
            if "layer" in vertex:
                member += "/" + vertex["layer"]
                if "unit" in vertex:
                    member += "/u" + str(vertex["unit"])
                elif member not in vertices:
                    vertices[member] = dict(kind="expert_layer", board=edge["board"])
            members.append(member)
        targets = [edge["board"] + "/" + e for e in edge["targets"]]
        vertices[identity] = dict(kind="hyperedge", **edge)
        relations.append(
            dict(
                kind="liquid_message",
                identity=identity,
                members=members + targets,
                sources=members,
                targets=targets,
                fields=edge["fields"],
                attention=edge["board"] + "/attention/" + edge["attention"],
                parents=edge["parents"],
            )
        )
    # Object identity proves operational parameter tying; a hash only proves equal content.
    for name, parameter in graph.named_parameters(remove_duplicate=False):
        canonical = parameter_owners.setdefault(id(parameter), name)
        raw = parameter.detach().cpu().contiguous()
        header = json.dumps(
            dict(dtype=str(raw.dtype), shape=list(raw.shape)), sort_keys=True
        ).encode()
        address = hashlib.sha256(
            header + raw.reshape(-1).view(__import__("torch").uint8).numpy().tobytes()
        ).hexdigest()
        vertices["tensor/" + name] = dict(
            kind="tensor",
            canonical="tensor/" + canonical,
            sha256=address,
            shape=list(raw.shape),
            dtype=str(raw.dtype),
        )
    return dict(
        schema="liquid-hypergraph-inspection-v1",
        profile=graph.profile(),
        vertices=vertices,
        relations=relations,
        layout=graph.layout(),
    )


def mermaid(graph):
    rows = ["flowchart LR"]

    def safe(value):
        return "n" + hashlib.sha256(value.encode()).hexdigest()[:12]

    for board, core in graph.cores.items():
        rows.append(f'  {safe(board)}["{board}: liquid board"]')
        for expert in core.ids:
            identity = board + "/" + expert
            rows += [
                f'  {safe(identity)}["{identity}"]',
                f"  {safe(board)} --> {safe(identity)}",
            ]
    for key, edge in graph.edges.items():
        rows.append(f"  {safe(key)} --> {safe(edge['child'])}")
    for identity, edge in graph.hyperedges.items():
        rows.append(f'  {safe(identity)}{{"{identity}: delta-gated liquid message"}}')
        for source in edge["sources"]:
            rows.append(f"  {safe(edge['board'] + '/' + source)} --> {safe(identity)}")
        for target in edge["targets"]:
            rows.append(f"  {safe(identity)} --> {safe(edge['board'] + '/' + target)}")
    return "\n".join(rows) + "\n"
