"""Neo4j Cypher retriever query"""

ARTICLE_RETRIEVAL_QUERY = """
// Collect initial search result nodes
WITH collect({node: node, score: score}) AS init_nodes_data
UNWIND init_nodes_data AS item

// Find related nodes
OPTIONAL MATCH (:Effective {id: item.node.id})-[:RELATED_TO|REFER_TO]-(related_node)
WITH init_nodes_data, collect(DISTINCT related_node) AS all_related_nodes
WITH init_nodes_data, [
     node IN all_related_nodes
     WHERE NOT node IN [item IN init_nodes_data | item.node]
] AS related_nodes

// Calculate vector similarity scores for related nodes
UNWIND related_nodes AS candidate_node
WITH init_nodes_data, candidate_node
WITH init_nodes_data, collect({
    node: candidate_node,
    score: vector.similarity.cosine($query_vector, candidate_node.embedding)
}) AS related_nodes_data

// Preserve initial nodes data for later
WITH init_nodes_data, related_nodes_data
UNWIND related_nodes_data AS related_item

// Sort only related items
WITH init_nodes_data, related_item
ORDER BY related_item.score DESC
LIMIT $limit

// Collect sorted related items
WITH init_nodes_data, collect(related_item) AS sorted_related_data

// Combine into a single list
WITH init_nodes_data + sorted_related_data AS all_nodes_data
UNWIND all_nodes_data AS data

// Return final results
RETURN data.node.text AS text, data.score AS score, {
    id: data.node.id
} AS metadata
"""

DEFINITION_RETRIEVAL_QUERY = """
RETURN node.text AS text, score, {
    id: node.id
} AS metadata
"""
