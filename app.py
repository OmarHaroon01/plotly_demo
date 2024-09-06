import random
import yfinance as yf
import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import dash_bootstrap_components as dbc
import os
from langchain_cohere import ChatCohere
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

os.environ["COHERE_API_KEY"] = "2zPsEcvQ1tzELqmw0ORVcVEX51H5p7KXLrpaqmrz"
model = ChatCohere(model="command-r-plus")
memory = ConversationBufferMemory(memory_key="messages", return_messages=True)
prompt = ChatPromptTemplate(
        input_variables=["content", "messages"],
        messages=[
            MessagesPlaceholder(variable_name="messages"),
            HumanMessagePromptTemplate.from_template("{content}")
        ]
    )
chain = LLMChain(llm=model, prompt=prompt, memory=memory)

symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "NFLX", "BABA", "INTC"]

symbol = random.choice(symbols)

external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = dbc.Container([
    html.Div([
        html.Div([
          dcc.Dropdown(symbols, "AAPL", id='dropdown-selection'),
          html.Div(
              dbc.Card([
                  dbc.CardBody([
                      html.Div(id='chat-history', className='d-flex flex-column flex-grow-1 overflow-auto'),
                      dcc.Textarea(
                          id='chat-input',
                          placeholder='Type a message...',
                          style={'width': '100%', 'height': '50px'}
                      ),
                      dbc.Button('Ask Chatbot', id='send-button', color='primary', className='mt-2', disabled=False),
                  ], className='d-flex flex-column h-100'),
              ], className='h-100'),
              className='flex-grow-1 pt-2 h-100'
          )
        ], className='d-flex flex-column h-100 col-12 col-md-6 p-1'),
        html.Div([
          dcc.Graph(id='graph-content'),
          html.Div(dbc.Table(id='table-content'), className='border border-2 border-dark overflow-auto')
        ], className='d-flex flex-column col-12 col-md-6 h-100 p-1')
    ], className='row h-100'),
], fluid=True, className='p-3', style={'min-height': '100vh', 'height': '100vh'})

@callback(
    [
      Output('graph-content', 'figure'),
      Output('table-content', 'children')
    ],
    Input('dropdown-selection', 'value')
)
def update_graph(value):
    global chain, prompt
    stock_details = yf.Ticker(value)
    stock_history = stock_details.history(period="1y")
    stock_news = stock_details.news

    # Process stock history for the graph
    updated_df = stock_history[['Close']].reset_index()
    updated_df['Date'] = updated_df['Date'].dt.date
    updated_df.rename(columns={'Close': 'Profit'}, inplace=True)
    figure = px.line(updated_df, x="Date", y="Profit")



    context = updated_df.to_string(index=False)
    system_message = f"""Use the following context to answer the user's question. Dont make any changes to the context.
                        Context: The current stock is {value}. {context}
                      """

    prompt = ChatPromptTemplate(
        input_variables=["content", "messages"],
        messages=[
            SystemMessage(system_message),
            MessagesPlaceholder(variable_name="messages"),
            HumanMessagePromptTemplate.from_template("{content}")
        ]
    )
    chain = LLMChain(llm=model, prompt=prompt, memory=memory)




    # Process stock news for the table
    news_df = pd.DataFrame(stock_news, columns=['title', 'link'])
    news_df.rename(columns={'title': 'Title', 'link': 'Link'}, inplace=True)
    news_table = dbc.Table.from_dataframe(news_df, striped=True, bordered=True, hover=True)

    return figure, news_table

@callback(
    Output('chat-history', 'children'),
    Output('chat-input', 'value'),
    Input('send-button', 'n_clicks'),
    State('chat-input', 'value'),
    prevent_initial_call=True
)
def update_chat(n_clicks, message):

    if not message:
        return html.Div(), ""
    
    
    result = chain({"content": message})
    response = result["messages"]
    chat_history = []
    for msg in response:
        if isinstance(msg, HumanMessage):
            chat_history.append(dcc.Markdown(f"**YOU:** {msg.content}"))
        elif isinstance(msg, AIMessage):
            chat_history.append(dcc.Markdown(f"**CHATBOT:** {msg.content}"))

    # Update chat history
    return html.Div(chat_history, className='mt-2'), ""

    

if __name__ == '__main__':
    app.run(debug=True)