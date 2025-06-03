ARTICLE_RETRIEVAL_QUERY_1 = """
// Inisialisasi node hasil pencarian awal
WITH collect({node: node, score: score}) AS init_nodes_data
UNWIND init_nodes_data AS item

// Temukan node terkait
OPTIONAL MATCH (:Effective {id: item.node.id})-[:RELATED_TO|REFER_TO]-(related_node)
WITH init_nodes_data, collect(DISTINCT related_node) AS all_related_nodes
WITH init_nodes_data, [
     node IN all_related_nodes
     WHERE NOT node IN [item IN init_nodes_data | item.node]
] AS related_nodes

// Hitung skor kemiripan vektor untuk node terkait
UNWIND related_nodes AS candidate_node
WITH init_nodes_data, candidate_node
WITH init_nodes_data, collect({
    node: candidate_node,
    score: vector.similarity.cosine($query_vector, candidate_node.embedding)
}) AS related_nodes_data

// Simpan init_nodes_data untuk nanti
WITH init_nodes_data, related_nodes_data
UNWIND related_nodes_data AS related_item

// Urutkan hanya related items
WITH init_nodes_data, related_item
ORDER BY related_item.score DESC
LIMIT $limit

// Gabungkan: pertama kumpulkan kembali yang related sudah diurutkan
WITH init_nodes_data, collect(related_item) AS sorted_related_data

// Gabungkan ke dalam satu list
WITH init_nodes_data + sorted_related_data AS all_nodes_data
UNWIND all_nodes_data AS data

// Kembalikan hasil akhir
RETURN data.node.text AS text, {
    id: data.node.id,
    type: "Article",
    source: data.node.source,
    score: data.score
} AS metadata
"""

# ARTICLE_RETRIEVAL_QUERY_1 = """
# WITH node, score

# RETURN node.text AS text, {
#     id: node.id,
#     type: "Article",
#     source: node.source,
#     score: score
# } AS metadata
# """

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
"""