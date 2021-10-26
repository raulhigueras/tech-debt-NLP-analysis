import streamlit as st
from utils import *

issue_type_list = ("Bug", "Dependency upgrade", "Documentation", "Epic", "Improvement", "New Feature", "Project",
                   "Question", "RTC", "Story", "Sub-task", "Task", "Technical task", "Test", "Umbrella", "Wish")


def get_topic_and_commonness(text):

    text_embedding = text_to_tf(text)

    topic, commonness = bow_to_topic(text_embedding)

    return topic, commonness


def get_difficulty(text, issue_type):

    text_embedding = text_to_svd_vector(text)

    difficulty = svd_vector_to_difficulty(text_embedding, issue_type)

    return difficulty


def main():

    # configuration
    st.title("Issue characterization based on descriptions")
    st.subheader("Results of the NLP analysis.")
    st.write("Authors: Jordi Puig, Raúl Higueras, Patricia Cabot")

    input_text = st.text_area("Description or summary of your issue")
    issue_type = st.selectbox("Select the issue type.", issue_type_list)

    if st.button("Run the models!"):

        if input_text is not "" or input_text is not " ":

            column1, column2 = st.columns(2)

            preprocess_text = raw_text_to_df(input_text)

            topic, commonness = get_topic_and_commonness(preprocess_text)
            difficulty = get_difficulty(preprocess_text, issue_type)

            column1.subheader("Topic classification and commonness:")
            column1.markdown(f"#### - Topic: {topic}")
            column1.markdown(f"#### - Commonness: {commonness}")

            column2.subheader("Difficulty associated:")
            column2.markdown(f"#### - Difficulty: {difficulty}")

            column1, column2 = st.columns(2)

            with column1.expander("Get information about the topics"):
                st.write("""
                Topic commonness:
                - ⭐️⭐️⭐️**Very common**: represents more than 40% of the issues.
                - ⭐️⭐**️️Common**: represents between the 5% and 40% of the issues. 
                - ⭐**Rare**: represents less than 5% of the issues. 
                """)

            with column2.expander("Get information about the difficulty"):
                st.write("""
                Issue difficulty metric: 
                - ❗**️Easy**: requires adding less than 15 lines in average. 
                - ❗️❗️**Medium**: requires adding between 15 and 115 lines in average. 
                - ❗️❗️❗**️Hard**: requires adding more than 115 lines in average. 
                """)
        else:
            st.write("You must enter an issue description or summary!")


if __name__ == "__main__":
    main()
