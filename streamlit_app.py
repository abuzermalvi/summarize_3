import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
from langchain_community.utilities import GoogleSerperAPIWrapper
from google import genai
import pickle
from io import BytesIO
import pytrends
from pytrends.request import TrendReq
from functools import reduce

# Set the API key for Google Serper
#os.environ["SERPER_API_KEY"] = "51d999e5b64d028cc0e05e4f27bc0fd40f70cfae"
#os.environ["GROQ_API_KEY"] = "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"
# Define the keywords for each brand
# Default keywords for each brand
DEFAULT_PG_KEYWORDS = {
    'Dawn': ['Dawn Dish Soap', 'Dawn Powerwash', 'Dawn Ultra'],
    'Cascade': ['Cascade Platinum', 'Cascade Platinum Plus', 'Cascade Complete', 'Cascade Dishwasher Pods'],
    'Swiffer': ['Swiffer WetJet', 'Swiffer PowerMop', 'Swiffer pads', 'Swiffer refills'],
    'Febreze': ['Febreze air freshener', 'Febreze fabric spray', 'Febreze plug in'],
    'Mr. Clean': ['Mr. Clean Magic Eraser', 'Mr. Clean all-purpose cleaner'],
    'Economic Factors': ['Tariffs', 'DEI', 'Latin Freeze', 'Recession', 'Immigration', 'Tax Refund']
}
# pg_keywords = {'Dawn':['Dawn Dish Soap', 'Dawn Powerwash', 'Dawn Ultra']}
trends_keywords = ['Tariffs', 'DEI', 'Latin Freeze', 'Recession', 'Immigration', 'Tax Refund']
# Function to collect news data for each keyword using the Google Serper API
def get_google_news_data(keywords):
    summary_dict = {}
    for keyword in keywords:
        search = GoogleSerperAPIWrapper(type="news")
        results = search.results(keyword)
        news_data = {
            'Title': [n.get('title', '') for n in results.get('news', [])],
            'Link': [n.get('link', '') for n in results.get('news', [])],
            'Date': [n.get('date', '') for n in results.get('news', [])],
            'Source': [n.get('source', '') for n in results.get('news', [])]
        }
        news_df = pd.DataFrame(news_data)
        news_df['Keyword'] = keyword
        news_df['Position'] = [i+1 for i in range(len(news_df))]
        summary_dict[keyword] = news_df
    return summary_dict

# Function to get the Gemini client for summarization
def get_gemini_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Function to extract a news summary using the Gemini API
def extract_news_summary(topic, time_posted):
    utc_time = datetime.utcnow()
    client = get_gemini_client()

    prompt = f"""You are an experienced journalist, with over 30 years of experience providing enriching and true news 
to viewers worldwide. Given an article topic, your task is to fetch all the information regarding the article and 
summarize it crisp and clear for the user. Also, analyze the information in detail and provide your view on whether 
the information has any potential impact on the CPG Industry. Return your response in the output format below.

Output Format:
<summary>
[The summary goes here]
</summary>
<impact_on_cpg_industry>
[Your view on the impact on the CPG industry goes here]
</impact_on_cpg_industry>

Instructions:
1. Ensure the response follows the output format.
2. Think step by step to arrive at the final response.
3. If the information does not impact the CPG industry, mention 'Not Applicable'.

Conversation Date: Today is {utc_time}
Time when article was posted: {time_posted}
User Article Topic: {topic}

Response:
"""
    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=prompt,
        config={"tools": [{"google_search": {}}]},
    )
    return response.text

# Function to retry extraction up to 3 times if needed
def extract_with_retry(topic, time_posted, retries=3):
    for attempt in range(1, retries + 1):
        try:
            response = extract_news_summary(topic, time_posted)
            if "<summary>" in response and "<impact_on_cpg_industry>" in response:
                return response
            else:
                st.write(f"Attempt {attempt}: Response format invalid. Retrying...")
        except Exception as e:
            st.write(f"Attempt {attempt} failed with error: {e}")
    raise Exception(f"Failed to extract news summary for topic '{topic}' after {retries} attempts")

from langchain_groq import ChatGroq
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
def call_llm_chain_base(base_prompt, parameters, variables):
    """Function for creating LLMChain instance

    Args:
        base_prompt (str): This is the base Prompt which describes the instructions
        parameters (Dict): This is a dict containing the LLM parameters
        variables (List): These are variables like questions/arguments that needs to be passed to the base prompt

    Returns:
        Callable: Returns the LLMChain that can be called for executing a prompt and getting response
    """
    # llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", **parameters)
    llm = ChatGroq(model='llama-3.3-70b-versatile', **parameters)
    prompt = PromptTemplate(input_variables=variables, template=base_prompt)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    return llm_chain

def concise_summary(summary, length):
    prompt= """
    You are a summary and information distillation expert. Given a text, your task is to summerize the text in less than 150 words.
    The summary should be in numeric bullet point format. Here us the output format:
    
    Output Format:
    <summary>
    1. bullet point 1 goes here
    2. bullet point 2 goes here
    .
    .
    n. bullet point n goes here
    </summary>
    
    Instructions:
    1. Make sure to thoroughly understand the text before generating the summary.
    2. If the text already appears to be less than 150 words, then do not summerizer it further and share back the same information in bullet points.
    3. Always make sure that the summary returned as response should be in numeric bullet points.
    4. Make sure to follow the output format while returning the response.
    5. Think step-by-step to approach the problem.
    
    Text:
    {text}
    
    Current Length of Text:
    {length} words
    
    Response:
"""
    params_dict = {"temperature":0.7}
    chain = call_llm_chain_base(prompt, params_dict, ["text", "length"])
    response = chain.run({"text": summary, "length": length})
    
    return response

def extract_with_retry_concise(summary, length, retries=3):
    for attempt in range(1, retries + 1):
        try:
            response = concise_summary(summary, length)
            if "<summary>" in response and "</summary>" in response:
                return response
            else:
                st.write(f"Attempt {attempt}: Response format invalid. Retrying...")
        except Exception as e:
            st.write(f"Attempt {attempt} failed with error: {e}")
    raise Exception(f"Failed to create concise summary after {retries} attempts")

# Function to process summaries for each keyword while showing a spinner
def process_summaries(summary_dict, keywords):
    summary_response_1st_topic = {}
    summary_response_2nd_topic = {}
    impact_response_1st_topic = {}
    impact_response_2nd_topic = {}
    if summary_response_1st_topic not in st.session_state:
        st.session_state['summary_response_1st_topic'] = {}
    if summary_response_2nd_topic not in st.session_state:
        st.session_state['summary_response_2nd_topic'] = {}
    if impact_response_1st_topic not in st.session_state:
        st.session_state['impact_response_1st_topic'] = {}
    if impact_response_2nd_topic not in st.session_state:
        st.session_state['impact_response_2nd_topic'] = {}
    for keyword in keywords:
        if keyword not in st.session_state['keyword_summary'] or st.session_state['keyword_summary'][keyword]==False:
            with st.spinner(f"Processing keyword: {keyword}"):
                # Use the first news article from the search results for each keyword
                title = summary_dict[keyword].loc[0, 'Title']
                date_posted = summary_dict[keyword].loc[0, 'Date']

                # Extract summary for the first topic
                response_1st_topic = extract_with_retry(title, date_posted, retries=3)
                pg_summary_1st_topic = response_1st_topic.split("<summary>")[1].split("</summary>")[0].strip()
                pg_impact_1st_topic  = response_1st_topic.split("<impact_on_cpg_industry>")[1].split("</impact_on_cpg_industry>")[0].strip()

                # Extract summary for the second topic
                response_2nd_topic = extract_with_retry(title, date_posted, retries=3)
                pg_summary_2nd_topic = response_2nd_topic.split("<summary>")[1].split("</summary>")[0].strip()
                pg_impact_2nd_topic  = response_2nd_topic.split("<impact_on_cpg_industry>")[1].split("</impact_on_cpg_industry>")[0].strip()

                pg_summary_1st_topic_concise = extract_with_retry_concise(pg_summary_1st_topic, len(list(pg_summary_1st_topic.split(" "))))
                pg_summary_2nd_topic_concise = extract_with_retry_concise(pg_summary_2nd_topic, len(list(pg_summary_2nd_topic.split(" "))))
                
                summary_response_1st_topic[keyword] = pg_summary_1st_topic_concise.split("<summary>")[1].split("</summary>")[0].strip()
                summary_response_2nd_topic[keyword] = pg_summary_2nd_topic_concise.split("<summary>")[1].split("</summary>")[0].strip()
                impact_response_1st_topic[keyword] = pg_impact_1st_topic
                impact_response_2nd_topic[keyword] = pg_impact_2nd_topic
                st.session_state.summary_response_1st_topic[keyword] = summary_response_1st_topic[keyword]
                st.session_state.summary_response_2nd_topic[keyword] = summary_response_2nd_topic[keyword]
                st.session_state.impact_response_1st_topic[keyword] = impact_response_1st_topic[keyword]
                st.session_state.impact_response_2nd_topic[keyword] = impact_response_2nd_topic[keyword]
                st.session_state['keyword_summary'][keyword] = True
            st.write(st.session_state.summary_response_2nd_topic.keys())
        else:
            st.write('ree')
            st.write(st.session_state.summary_response_1st_topic.keys())
            summary_response_1st_topic[keyword] = st.session_state.summary_response_1st_topic[keyword]
            summary_response_2nd_topic[keyword] = st.session_state.summary_response_2nd_topic[keyword]
            impact_response_1st_topic[keyword] = st.session_state.impact_response_1st_topic[keyword]
            impact_response_2nd_topic[keyword] = st.session_state.impact_response_2nd_topic[keyword]
            st.write('gg')
    return summary_response_1st_topic, summary_response_2nd_topic, impact_response_1st_topic, impact_response_2nd_topic

def get_trends_data(keywords):
    try:
        pytrends = TrendReq(hl='en-US', tz=360, retries=10, backoff_factor=0.1)
        pytrends.build_payload(keywords, cat=0, timeframe='today 12-m', geo='US', gprop='news')
        data = pytrends.interest_over_time()
        return data
    except Exception as e:
        raise Exception(f"Failed to extract trends data for keywords '{keywords}' after 10 attempts")

# Function to run the complete processing and return final dataframes
def run_process(pg_keywords):
    if 'news_overall_df' not in st.session_state:
        st.session_state['news_overall_df'] = None
    if 'summary_overall_df' not in st.session_state:
        st.session_state['summary_overall_df'] = None
    if 'pg_keywords' not in st.session_state:
        st.session_state['pg_keywords'] = None
    if 'keyword_summary' not in st.session_state:
        st.session_state['keyword_summary'] = {}
    if 'google_trends_data' not in st.session_state:
        st.session_state['google_trends_data'] = None
    st.write(st.session_state['keyword_summary'])
    if st.session_state['pg_keywords']!= pg_keywords or st.session_state['pg_keywords'] is None or not all(st.session_state['keyword_summary'].values()):
        news_overall_dict = {}
        summary_overall_dict = {}
        # Process for each brand and its list of keywords
        for brand, keyword_list in pg_keywords.items():
            # Get news data for the keywords
            news_dict = get_google_news_data(keyword_list)
            st.write("Reached Here!")
            news_final = pd.concat([df for df in news_dict.values()], axis=0, ignore_index=True)
            news_final['Brand'] = brand

            # Process summaries using the helper function (with spinners)
            summary_response_1st_topic, summary_response_2nd_topic, impact_response_1st_topic, impact_response_2nd_topic = process_summaries(news_dict, keyword_list)
            st.write(summary_response_1st_topic.keys())
            df_summaries = pd.DataFrame({
                "Brand": [brand] * len(keyword_list),
                "Keyword": keyword_list,
                "First Topic Summary": [summary_response_1st_topic[key] for key in keyword_list],
                "Second Topic Summary": [summary_response_2nd_topic[key] for key in keyword_list],
                "First Topic Impact": [impact_response_1st_topic[key] for key in keyword_list],
                "Second Topic Impact": [impact_response_2nd_topic[key] for key in keyword_list],
            })
            news_overall_dict[brand] = news_final
            summary_overall_dict[brand] = df_summaries

        news_overall_df = pd.concat([df for df in news_overall_dict.values()], axis=0, ignore_index=True)
        summary_overall_df = pd.concat([df for df in summary_overall_dict.values()], axis=0, ignore_index=True)
        st.session_state['pg_keywords'] = pg_keywords
        st.session_state['news_overall_df'] = news_overall_df
        st.session_state['summary_overall_df'] = summary_overall_df
    if st.session_state['google_trends_data'] is None:
      try:
        keyword_list = [trends_keywords[i:i+3] for i in range(0, len(trends_keywords), 3)]
        final_trends_list = []
        for i in keyword_list:
            trends_df = get_trends_data(i)
            final_trends_list.append(trends_df.drop(columns=['isPartial'], axis=1).reset_index())
        st.session_state['google_trends_data'] = reduce(lambda left, right: pd.merge(left, right, on='date', how='inner'), final_trends_list)
      except:
        st.session_state['google_trends_data'] = None
    return st.session_state['news_overall_df'], st.session_state['summary_overall_df'], st.session_state['google_trends_data']

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.close()
    processed_data = output.getvalue()
    return processed_data

# Main Streamlit app
def main():
    st.title("News and Summary Data Processing")
    
    # Initialize session state
    if 'process_completed' not in st.session_state:
        st.session_state.process_completed = False
    if 'pg_keywords_selected' not in st.session_state:
        st.session_state.pg_keywords_selected = DEFAULT_PG_KEYWORDS.copy()
    if 'keyword_summary' not in st.session_state:
        st.session_state.keyword_summary = {}

    # Keyword selection form to avoid reruns on change
    with st.form(key='keyword_selection_form'):
        st.subheader("Select Keywords for Each Brand")
        selected = {}
        for brand, options in DEFAULT_PG_KEYWORDS.items():
            selected_list = st.multiselect(
                label=f"{brand}",
                options=options,
                default=st.session_state.pg_keywords_selected.get(brand, []),
                key=f"sel_{brand}"
            )
            selected[brand] = selected_list

        submit = st.form_submit_button(label='Update Keywords')

    if submit:
        # Update session_state once on form submit
        st.session_state.pg_keywords_selected = {b: selected[b] for b in selected if selected[b]}
        st.success("Keyword selections updated. Click Start Process to run with new keywords.")
    
    # Button to trigger the initial process
    if st.button("Start Process"):
        try:
            news_df, summary_df, trends_df = run_process(st.session_state.pg_keywords_selected)
            st.session_state.process_completed = True  # Mark process as completed
            st.success("Process completed successfully!")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Display data and rerun options only if process was completed
    if st.session_state.process_completed:
        st.subheader("News Overall DataFrame")
        st.dataframe(st.session_state['news_overall_df'])
        
        st.subheader("Summary Overall DataFrame")
        st.dataframe(st.session_state['summary_overall_df'])
        
        st.subheader("Trends Data")
        st.dataframe(st.session_state['google_trends_data'])
        
        st.download_button(label="Download News Data", data=to_excel(st.session_state['news_overall_df']), file_name="df_news.xlsx", mime="xlsx")
        st.download_button(label="Download Article Summary", data=to_excel(st.session_state['summary_overall_df']), file_name="df_summary.xlsx", mime="xlsx")
        if st.session_state['google_trends_data']:
          st.download_button(label="Download Trends", data=to_excel(st.session_state['google_trends_data']), file_name="df_trends.xlsx", mime="xlsx")
        
        # # Rerun section
        # keyword_rerun = st.selectbox(
        #     "Select the keyword to rerun summary generation", 
        #     st.session_state['keyword_summary'].keys()
        # )
        # if st.button("Rerun Summary Generation for the keyword"):
        #     # Update the session state for the selected keyword
        #     st.session_state['keyword_summary'][keyword_rerun] = False
        #     st.rerun()  # Force a rerun to trigger the process again

if __name__ == "__main__":
    main()
