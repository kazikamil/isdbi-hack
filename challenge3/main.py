from langgraph.graph import Graph, START,END
from reviewer_agent import ReviewerAgent
from enhancer_agent import EnhancerAgent
from validator_agent import ValidatorAgent
# Initialiser le graphe
graph = Graph()

# Ajouter les noeuds
graph.add_node("reviewer", ReviewerAgent())
graph.add_node("enhancer", EnhancerAgent())
graph.add_node("validator", ValidatorAgent())

# Configurer les connections EXPLICITEMENT
graph.add_edge(START, "reviewer")  # 🔑 Connection critique
graph.add_edge("reviewer", "enhancer")
graph.add_edge("enhancer", "validator")
graph.add_edge("validator", END)    # Si utilisation d'un point final

# Compiler
app = graph.compile()
text = input("🔍 Enter AAOIFI : ")
# Exécuter
result = app.invoke({
    "text": text
})

print("-----------------------Validation-----------------------")
print(result)