from langgraph.graph import Graph, START,END
from extractor import ExtractorAgent
from matcher import MatcherAgent
from compliance_validator import ValidAgent
# Initialiser le graphe
graph = Graph()

# Ajouter les noeuds
graph.add_node("extractor", ExtractorAgent())
graph.add_node("matcher", MatcherAgent())
graph.add_node("compliance", ValidAgent())

# Configurer les connections EXPLICITEMENT
graph.add_edge(START, "extractor")  # 🔑 Connection critique
graph.add_edge("extractor", "matcher")
graph.add_edge("matcher", "compliance")
graph.add_edge("compliance", END)    # Si utilisation d'un point final

# Compiler
app = graph.compile()
text = input("🔍 the lease : ")
# Exécuter
result = app.invoke({
    "text": text
})

print("-----------------------Validation-----------------------")
print(result['compliance'])
