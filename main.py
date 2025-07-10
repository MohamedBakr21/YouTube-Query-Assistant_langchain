import streamlit as st
import app as langg 

st.title("Youtube assistant")

# Get YouTube video URL
youtube_url = st.text_area("Enter the URL of the video:")

# Get query input
query = st.text_input("Enter your query:")

# Button to generate response
if st.button("Generate"):
    if not youtube_url.strip() or not query.strip():
        st.warning("Please enter both a video URL and a query.")
    else:
        with st.spinner("Processing..."):
            # Get document from YouTube
            doc_vectorstore = langg.get_doc_from_youtube(youtube_url)

            # Use similarity search to get relevant docs
            relevant_docs = doc_vectorstore.similarity_search(query, k=3)

            # Combine retrieved docs into a single context string
            context = "\n".join([doc.page_content for doc in relevant_docs])

            # Get response from the LLM
            response = langg.get_response(query, context)

            # Display the response
            st.subheader("Answer:")
            st.write(response)
