import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(array):

    rng = np.random.RandomState(10)  # deterministic random data
    a = np.hstack((rng.normal(size=1000),
                rng.normal(loc=5, scale=2, size=1000)))
    _ = plt.hist([len(x) for x in array], bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()

def plot_curves(trace_train, trace_val):
    plt.plot(trace_train, label='train')
    plt.plot(trace_val, label='validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_dataframe(df):
    # https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format
    import matplotlib.pyplot as plt
    import pandas as pd
    # from pandas.table.plotting import table # EDIT: see deprecation warnings below
    from pandas.plotting import table 

    ax = plt.subplot(111, frame_on=False) # no visible frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis

    table(ax, df)  # where df is your data frame

    plt.savefig('mytable.png')