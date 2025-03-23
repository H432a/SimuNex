from langchain_groq import ChatGroq

llm = ChatGroq(
    temperature=0,
    groq_api_key="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",  # Replace this
    model="llama3-70b-8192"  # or whatever latest variant name they use in prod
)

def suggest_projects(detected_components):
    if not detected_components:
        return "⚠️ No components detected."

    prompt = f"The detected components are {', '.join(detected_components)}. Suggest innovative electronics or IoT projects using these components. Include 3 project ideas with short descriptions."
    response = llm.invoke(prompt)
    return response.content.strip()
