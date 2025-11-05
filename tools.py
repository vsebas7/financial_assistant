import json
import requests
from datetime import datetime
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

SECTORS_API_KEY = st.secrets["SECTORS_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


def retrieve_from_endpoint(url: str) -> dict:
    """
    A robust, reusable helper function to perform GET requests.
    """
    
    headers = {"Authorization": SECTORS_API_KEY}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()

        return data

    except requests.exceptions.HTTPError as err:
        return {
            "error": f"HTTPError {err.response.status_code} - {err.response.reason}",
            "url": url,
            "detail": err.response.text
        }
    
    except Exception as e:
        return {
            "error": f"Unexpected error: {type(e).__name__} - {str(e)}",
            "url": url
        }


@tool
def get_company_overview(stock: str) -> dict:
    """
    Get company overview
    
    @param stock: The stock symbol of the company
    @return: The company overview
    """

    url = f"https://api.sectors.app/v1/company/report/{stock}/?sections=overview"

    return retrieve_from_endpoint(url)


@tool
def get_company_revenue_cost_segments(stock : str) -> dict :
    """
    Return revenue and cost segments of a given stock.

    @param stock: The stock symbol of the company
    @return: The company revenue and cost segments
    """

    url = f"https://api.sectors.app/v1/company/get-segments/{stock}/"

    return retrieve_from_endpoint(url)

@tool
def get_top_companies_by_tx_volume(start_date: str, end_date: str, top_n: int = 5) -> dict:
    """
    Get top companies by transaction volume

    @param start_date: The start date in YYYY-MM-DD format
    @param end_date: The end date in YYYY-MM-DD format
    @param top_n: Number of stocks to show
    @return: A list of most traded IDX stocks based on transaction volume for a certain interval
    """
    url = f"https://api.sectors.app/v1/most-traded/?start={start_date}&end={end_date}&n_stock={top_n}"

    return retrieve_from_endpoint(url)

@tool
def get_daily_tx(stock: str, start_date: str, end_date: str) -> list[dict]:
    """
    Get daily transaction for a stock

    @param stock: The stock 4 letter symbol of the company
    @param start_date: The start date in YYYY-MM-DD format
    @param end_date: The end date in YYYY-MM-DD format
    @return: Daily transaction data of a given ticker for a certain interval
    """
    url = f"https://api.sectors.app/v1/daily/{stock}/?start={start_date}&end={end_date}"

    return retrieve_from_endpoint(url)


@tool
def get_top_companies_ranked(dimension: str, top_n: int, year: int) -> list[dict]:
    """
    Return a list of top companies (symbol) based on certain dimension 
    (dividend yield, total dividend, revenue, earnings, market cap,...)

    @param dimension: The dimension to rank the companies by, one of: 
    "dividend_yield", "total_dividend", "revenue", "earnings", "market_cap", ...

    @param top_n: Number of stocks to show
    @param year: Year of ranking, always show the most recent full calendar year that has ended
    @return: A list of top tickers in a given year based on certain classification
    """

    url = f"https://api.sectors.app/v1/companies/top/?classifications={dimension}&n_stock={top_n}&year={year}"

    return retrieve_from_endpoint(url)

@tool
def get_top_companies_by_growth(dimension : str, sub_sectors : str) -> dict :
    """
    Return a list of top companies (symbol) based on certain dimension 
    (top_earnings_growth_gainers, top_earnings_growth_losers, top_revenue_growth_gainers, top_revenue_growth_losers,...)

    @param dimension : The dimension to rank the companies by, one of: 
    top_earnings_growth_gainers, top_earnings_growth_losers, top_revenue_growth_gainers, top_revenue_growth_losers.
    @param sub_sectors : use get_company_overview tools to get the subsectors of the company, if not provided just leave it blank
    """

    url = f"https://api.sectors.app/v1/companies/top-growth/?classifications={dimension}&n_stock=5&sub_sector={sub_sectors}"

    return retrieve_from_endpoint(url)

@tool
def get_top_companies_by_mover(dimension : str, period : str, sub_sectors : str) -> dict :
    """
    Return a list of top companies (symbol) based on certain dimension on certain period
    (top_gainers, top_losers,...)

    @param dimension : The dimension to rank the companies by, one of: 
    (top_gainers, top_losers)
    @param period : The certain period, one of:
    (1d, 7d, 14d, 30d, 365d)
    @param sub_sectors : use get_company_overview tools to get the subsectors of the company, if not provided just leave it blank
    """

    url = f"https://api.sectors.app/v1/companies/top-changes/?classifications={dimension}&n_stock=5&periods={period}&sub_sector={sub_sectors}"

    return retrieve_from_endpoint(url)

def get_finance_agent():

    # Defined Tools
    tools = [
        get_company_overview,
        get_top_companies_by_tx_volume,
        get_daily_tx,
        get_top_companies_ranked,
        get_top_companies_by_growth,
        get_top_companies_by_mover,
        get_company_revenue_cost_segments
    ]

    # Create the Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
                Answer the following queries, being as factual and analytical as you can. 
                If you need the start and end dates but they are not explicitly provided, 
                infer from the query. Whenever you return a list of names, return also the 
                corresponding values for each name. If the volume was about a single day, 
                the start and end parameter should be the same. Note that the endpoint for 
                performance since IPO has only one required parameter, which is the stock. 
                Today's date is {datetime.today().strftime("%Y-%m-%d")}
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # Initializing the LLM
    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
    )

    # Create the Agent and AgentExecutor
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Add Memory to the AgentExecutor
    def get_session_history(session_id: str):

        return StreamlitChatMessageHistory(key=session_id)
    
    agent_with_memory = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return agent_with_memory