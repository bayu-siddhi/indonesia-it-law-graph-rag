ARTICLE_RETRIEVAL_QUERY_1 = """
// Inisialisasi node hasil pencarian awal
WITH collect({node: node, score: score}) AS init_nodes_data
UNWIND init_nodes_data AS item

// Temukan node terkait
OPTIONAL MATCH (:Effective {id: item.node.id})-[:RELATED_TO|REFER_TO]-(related_node)
WITH init_nodes_data, collect(DISTINCT related_node) AS all_related_nodes
WITH init_nodes_data, [
     node IN all_related_nodes
     WHERE NOT node IN [item IN init_nodes_data | item.node.id]
] AS related_nodes

// Hitung skor kemiripan vektor untuk node terkait
UNWIND related_nodes AS candidate_node
WITH init_nodes_data, candidate_node
WITH init_nodes_data, collect({
    node: candidate_node,
    score: vector.similarity.cosine($query_vector, candidate_node.embedding)
}) AS related_nodes_data

// Gabungkan hasil awal dan hasil pencarian node terkait
WITH init_nodes_data + related_nodes_data AS all_nodes_data
UNWIND all_nodes_data AS data
ORDER BY data.score DESC
LIMIT $limit

// Kembalikan hasil akhir
RETURN data.node.text AS text, {
    id: data.node.id,
    type: "Article",
    source: data.node.source,
    score: data.score
} AS metadata
"""

ARTICLE_RETRIEVAL_QUERY_2 = """
WITH node, score
WHERE NOT node.id IN $excluded_ids

RETURN node.text AS text, {
    id: node.id,
    type: "Article",
    source: node.source,
    score: score
} AS metadata

ORDER BY score DESC
LIMIT $limit
"""

DEFINITION_RETRIEVAL_QUERY = """
RETURN node.text AS text, {
    id: node.id,
    type: "Definition",
    source: node.source,
    score: score
} AS metadata

ORDER BY score DESC
LIMIT $limit
"""