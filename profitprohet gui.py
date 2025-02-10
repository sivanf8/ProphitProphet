import tkinter as tk
from tkinter import ttk,PhotoImage, messagebox
from tkinter.constants import END
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.losses.pytorch import MAE
from sklearn import metrics
import certifi
import json
import os
import sys
from datetime import timedelta
from urllib.request import urlopen
##################################################################################################################################################
###################   start functions here   ##############################

# -------- Functions
def reverse_data_by_dates(df):
    data = {'date': [], 'close': []}
    for i in df['date'].iloc[-1::-1]:
        data['date'].append(i)
    for j in df['close'].iloc[-1::-1]:
        data['close'].append(j)
    df1 = pd.DataFrame(data)
    df1["date"] = pd.to_datetime(df1["date"])
    return df1


def resampled_data_7days(df):
    df.set_index('date', inplace=True)
    date_range = pd.date_range(start=df.index.min(), end=df.index.max())
    df_resampled = df.reindex(date_range)
    df_resampled['close'] = df_resampled['close'].ffill()
    df_resampled.reset_index(inplace=True)
    df_resampled.rename(columns={'index': 'date'}, inplace=True)
    return df_resampled


def fit_df_to_model(df, unique_id):
    df["unique_id"] = unique_id
    df.columns = ["ds", "y", "unique_id"]
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def downsample_train_data(df, d):
    H = len(df) % d
    df = df[H:]

    df.set_index('date', inplace=True)  # Set 'date' as the index for resampling
    df = df.resample(str(d) + 'D').mean()  # Choose mean or any other aggregation method
    df.reset_index(inplace=True)
    return df


def downsample_test_data(df, d):
    df.set_index('date', inplace=True)  # Set 'date' as the index for resampling
    df = df.resample(str(d) + 'D').mean()  # Choose mean or any other aggregation method
    df.reset_index(inplace=True)
    return df


def downsample_with_alignment(df, d):
    df.set_index('date', inplace=True)
    last_date = df.index[-1]
    df_resampled = df.resample(f'{d}D', origin=last_date).mean()
    df_resampled.reset_index(inplace=True)
    return df_resampled


def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)


def df_stock(stock_symbol):
    try:
        url = (
                    "https://financialmodelingprep.com/api/v3/historical-price-full/" + stock_symbol + "?apikey=2f1c3c73de79230c2b4b038861ddf970")
        j = get_jsonparsed_data(url)
        df = pd.DataFrame(j['historical'])
    except KeyError:
        return 0
    return df.iloc[:, [0, 4]]

def Check_Best_Downsample_Factor(H, df):
    D = list(range(1, H + 1, 1))
    min_mae = float("inf")
    min_d = 0

    data_size = 700  # num of data
    p = 0.15  # percent of test\val data from the entire data

    test_size = math.ceil(p * data_size)
    train_size = data_size - test_size


    for k in D:
        progress_var.set(k)
        progress_label.config(text=f"     Processing {k} / {H}     ")
        app.update_idletasks()
        df1 = df.copy()
        df_len = len(df1)
        df_train = df.iloc[df_len - data_size:df_len - test_size]
        df_test = df.iloc[df_len - test_size:]

        # Downsample data
        df_train_downsample = downsample_train_data(df_train, k)
        df_test_downsample = downsample_test_data(df_test, k)

        # Fit to model
        df_train_vars = fit_df_to_model(df_train_downsample, "1")
        df_test_vars = fit_df_to_model(df_test_downsample, "1")

        # --- TRAIN
        H_match = math.floor(H / k)

        lstm_model = LSTM(h=H_match,
                          input_size=200,
                          loss=MAE(),
                          scaler_type="robust",
                          encoder_n_layers=2,
                          decoder_layers=2,
                          decoder_hidden_size=128,
                          encoder_hidden_size=128,
                          context_size=25,
                          max_steps=45,
                          early_stop_patience_steps=30
                          )

        nf = NeuralForecast(models=[lstm_model], freq=str(k) + 'D')

        # Validation Stage
        val_size = math.floor(test_size / k)
        nf.fit(df=df_train_vars, val_size=val_size)  # test_size=test_size

        # Prediction
        Y_forecast = nf.predict(futr_df=df_test_vars[0:H_match])  # step_size
        Y_forecast = Y_forecast.reset_index(drop=False).drop(columns=['unique_id'])

        # Calculate MAE
        forecast_data = np.array(Y_forecast['LSTM'])
        real_data = np.array(df_test_vars['y'][0:H_match])
        mae = metrics.mean_absolute_error(forecast_data, real_data)

        if (min_mae > mae):
            min_mae = mae
            min_d = k

    return min_d, round(100 - min_mae, 2)

# final code:
def forecast(H, stock_name):
    if len(stock_name)==0 and len(H)==0:
        messagebox.showerror("Error", "Error: Enter input !!")
        return
    elif len(stock_name)==0:
        messagebox.showerror("Error", "Error: Please insert a Stock Ticker !!")
        return

    stock_name=str(stock_name)
    try:
        H=int(H)
    except ValueError:
        messagebox.showerror("Error", "Error: Please insert a number for H !!")
        return
    data_size = 700  # num of data
    p = 0.15  # percent of test\val data from the entire data

    # Initialize the progress bar
    progress_var.set(0)
    progress_bar["maximum"] = H+1  # Total number of steps
    app.update_idletasks()

    # Retrive data
    df = df_stock(stock_name)
    if isinstance(df,int):
        messagebox.showerror("Error", "Error: Stock Ticker doesn't exist !!")
        return
    df = df[0:data_size]
    df = resampled_data_7days(reverse_data_by_dates(df))




    # Best D
    d, valid_per = Check_Best_Downsample_Factor(H, df)
    progress_var.set(H+1)
    progress_label.config(text=f" Processing finished!! ")
    app.update_idletasks()

    df_train = df.iloc[-data_size:]
    df_train_downsample = downsample_with_alignment(df_train, d)

    # Fit to model
    df_train_vars = fit_df_to_model(df_train_downsample, "1")

    # --- TRAIN
    H_match = math.floor(H / d)

    lstm_model = LSTM(h=H_match,
                      input_size=200,  # math.floor(15 / k)
                      loss=MAE(),
                      scaler_type="robust",
                      encoder_n_layers=2,
                      decoder_layers=2,
                      decoder_hidden_size=128,
                      encoder_hidden_size=128,
                      context_size=25,
                      max_steps=45,
                      early_stop_patience_steps=30
                      )

    nf = NeuralForecast(models=[lstm_model], freq=str(d) + 'D')

    # Validation Stage
    val_size = math.floor(p * data_size / d)
    nf.fit(df=df_train_vars, val_size=val_size)

    # Prediction
    future_df = pd.DataFrame({
        'unique_id': df_train_vars['unique_id'].unique()[0],
        'ds': pd.date_range(start=df_train_vars['ds'].max(), periods=H_match + 1, freq=f"{d}D")[1:]
    })
    Y_forecast = nf.predict(futr_df=future_df)
    Y_forecast = Y_forecast.reset_index(drop=False).drop(columns=['unique_id'])

    # ---------------------------round LSTM--------------------------------------------
    Y_forecast['LSTM'] = Y_forecast['LSTM'].astype(float)
    Y_forecast['LSTM'] = np.round(Y_forecast['LSTM'], 2)
    # ---------------------------------------------------------------------------------
    Y_forecast.columns=['Date','Close']
    Y_forecast['Date']=pd.to_datetime(Y_forecast['Date']).dt.date
    last_day = df_train_vars["ds"].iloc[-1].date()

    # return [d, valid_per, Y_forecast, last_day]

    # editing the app with the results:
    #### LAbels on right side ####
    tk.Label(app, text='Last Day Before Prediction: ' + str(last_day), font=('Arial', 11)).place(x=920, y=100)
    tk.Label(app, text='Last Day Close: ' + str(round(df_train_vars['y'].iloc[-1],2)), font=('Arial', 11)).place(x=920, y=120)
    tk.Label(app, text='Accuracy: ' + str(valid_per) + '%', font=('Arial', 11)).place(x=920, y=140)
    tk.Label(app, text='Chosen Downsample Factor: ' + str(d) , font=('Arial', 11)).place(x=920, y=160)

    #### Y hat DF window ####
    frame = tk.Frame(app)
    frame.place(x=10, y=100)
    tree = tk.ttk.Treeview(frame, columns=list(Y_forecast.columns), show='headings', height=23)
    tree.pack(side='left')
    scrollbar = tk.ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    scrollbar.pack(side='right', fill='y')
    tree.configure(yscrollcommand=scrollbar.set)
    for col in Y_forecast.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)
    for index, row in Y_forecast.iterrows():
        tree.insert("", "end", values=list(row))

    #### graph for stock ####
    #canvas = tk.Canvas(app, width=670, height=485, highlightthickness=0,bg='black')
    # canvas = ttk.FigureCanvasTkAgg(,
    #                            master=app)
    # canvas.draw()
    # canvas.place(x=240, y=100)
    Y_forecast.columns = ['ds', 'y']
    df_train_vars=df_train_vars.reset_index(drop=True).drop(columns=['unique_id'])
    plot_df=pd.concat([df_train_vars,Y_forecast],ignore_index=True)
    plot_df['ds'] = pd.to_datetime(plot_df['ds']).dt.date
    fig=plt.figure()
    plt.plot(plot_df['ds'],plot_df['y'])
    canvas = FigureCanvasTkAgg(fig,master = app)
    # canvas.place(x=240, y=100)
    canvas.draw()
    canvas.get_tk_widget().place(x=240, y=100)
    toolframe=tk.Frame(app)
    toolbar = NavigationToolbar2Tk(canvas,toolframe)
    toolframe.place(x=900,y=540)
    return
###################   end functions here   ##############################
##################################################################################################################################################

# df = pd.DataFrame(data)
def clear_text():
    for widget in app.winfo_children():
        if isinstance(widget,tk.Entry):
            widget.delete(0, END)
        if isinstance(widget,tk.Frame):
            widget.destroy()
        if isinstance(widget,tk.Canvas) and widget not in t_lst:
            widget.destroy()
        if isinstance(widget,tk.Label) and widget not in t_lst:
            widget.destroy()
        progress_var.set(0)
        progress_label.config(text="Waiting for input...")
        app.update_idletasks()
    return

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)



app= tk.Tk()
style = ttk.Style()
app.geometry("1200x600")
app.title(" ProfitProphet")
bg = PhotoImage(file = resource_path("image_bg.png"))
background_label = tk.Label(app, image=bg)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
can= tk.Canvas(app, width=860, height=50, highlightthickness=0)
can.place(x=0, y=0)
can.create_rectangle(5, 5, 860, 50,  outline="black", width=2)


#Progressbar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(app, variable=progress_var, maximum=5, length=300)
progress_bar.place(x=870, y=5)
progress_label = tk.Label(app, text="Waiting for input...", font=('Arial', 11))
progress_label.place(x=960, y=30)

t1=tk.Label(app, text='Stock Ticker:', font=('Arial', 13))
t1.place(x=8,y=15)
t2=tk.Label(app, text='H (Prediction Window Length):', font=('Arial', 13))
t2.place(x=288,y=15)
t3=tk.Label(app, text='Notice! The prediction is for trading week (5 days a week) and NOT for whole 7-days week!!', font=('Arial', 10),bg='yellow')
t3.place(x=8,y=60)
t_lst=[t1,t2,t3,can,background_label,progress_label]
e1 = tk.Entry(app,width=15,font=('Arial 15'))
e2 = tk.Entry(app,width=15,font=('Arial 15'))
e1.place(x=108,y=15)
e2.place(x=518,y=15)
button=tk.Button(app,command=lambda:forecast(e2.get(), e1.get()), text='Apply',font='Arial 11',width=6,bg='light green').place(x=698,y=13)
buttonreset=tk.Button(app,command=lambda:clear_text(), text='Clear Data',font='Arial 11',width=8,bg='#FF8080').place(x=770,y=13)
style.configure("Treeview.Heading", font=('Arial', 10, 'bold'))
app.iconbitmap(resource_path("growth-chart-invest.ico"))


app.mainloop()