import streamlit as st

# Set page title
st.title("Hello, Streamlit!")

# Display introductory text
st.write("This is a simple Streamlit app.")

# Add a slider
slider_val = st.slider("Select a number", 0, 100)
st.write(f"You selected: {slider_val}")

# Add a button
if st.button("Click me"):
    st.write("Button clicked!")
