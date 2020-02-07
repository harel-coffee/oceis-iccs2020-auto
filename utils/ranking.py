from scipy import stats
import pandas as pd
import os
import numpy as np
import plotly.plotly
import plotly.graph_objs as go
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt


class Ranking:

    def __init__(self, method_names=None, stream_names=None, test_name="wilcoxon", metrics=None, methods_alias=None, metrics_alias=None):
        self.method_names = method_names
        self.stream_names = stream_names
        # self.dimension = len(self.method_names)
        self.metrics = metrics
        self.test_name = test_name
        self.date_time = "{:%Y-%m_%d-%H-%M}".format(datetime.now())
        self.methods_alias = methods_alias
        self.metrics_alias = metrics_alias

    def test_sum(self, treshold=0.05, auto_open=True):

        data = {}
        ranking = {}
        self.iter = 0

        for method_name in self.method_names:
            ranking[method_name] = 0
            for stream_name in self.stream_names:
                try:
                    data[(method_name, stream_name)] = pd.read_csv("results/raw_%s/%s/%s.csv" % (self.experiment_name, stream_name, method_name), header=0, index_col=0)
                except:
                    print("None is ", method_name, stream_name)
                    data[(method_name, stream_name)] = None

        if self.metrics is None:
            self.metrics = data[(self.method_names[0], self.stream_names[0])].columns.values

        for stream in self.stream_names:
            for metric in self.metrics:
                if self.test_name is "tstudent":
                    for i, method_1 in enumerate(self.method_names):
                        for j, method_2 in enumerate(self.method_names):
                            if method_1 == method_2:
                                continue
                            if data[(method_2, stream_name)] is None:
                                ranking[method_1] += 1
                                print("if1", method_2)
                                self.iter += 1
                                continue
                            if data[(method_1, stream_name)] is None:
                                print("if2", method_1)
                                continue

                            self.iter += 1
                            try:
                                statistic, p_value = stats.ttest_ind(data[(method_1, stream)][metric].values, data[(method_2, stream)][metric].values)
                            except:
                                print(method_1, method_2)
                            if p_value < treshold and statistic > 0:
                                ranking[method_1] += 1
                elif self.test_name is "wilcoxon":
                    for i, method_1 in enumerate(self.method_names):
                        for j, method_2 in enumerate(self.method_names):
                            if method_1 == method_2:
                                continue
                            if data[(method_2, stream_name)] is None:
                                ranking[method_1] += 1
                                print("if1", method_2)
                                self.iter += 1
                                continue
                            if data[(method_1, stream_name)] is None:
                                print("if2", method_1)
                                continue

                            self.iter += 1
                            try:
                                statistic, p_value = stats.ranksums(data[(method_1, stream)][metric].values, data[(method_2, stream)][metric].values)
                                if p_value < treshold:
                                    ranking[method_1] += statistic
                            except:
                                print(method_1, method_2)

        if not os.path.exists("results/ranking_tests/%s/" % (self.date_time)):
            os.makedirs("results/ranking_tests/%s/" % (self.date_time))

        trace = self.prepare_trace(ranking)
        layout = go.Layout(title='Ranking %s tests summarise' % (self.test_name), plot_bgcolor='rgb(230, 230, 230)')
        fig = dict(data=[trace], layout=layout)
        plotly.offline.plot(fig, filename="results/ranking_tests/%s/ranking_sum_%s.html" % (self.date_time, self.test_name), auto_open=auto_open)

    def test_streams(self, treshold=0.001, auto_open=True):
        data = {}
        ranking = {}
        for method_name in self.method_names:
            ranking[method_name] = 0
            for stream_name in self.stream_names:
                data[(method_name, stream_name)] = pd.read_csv("results/raw/%s/%s.csv" % (stream_name, method_name), header=0, index_col=0)

        if not self.metrics:
            self.metrics = data[(self.method_names[0], self.stream_names[0])].columns.values

        for stream in self.stream_names:
            ranking = {}
            self.iter = 0
            for method_name in self.method_names:
                ranking[method_name] = 0
            for metric in self.metrics:
                if self.test_name is "tstudent":
                    for i, method_1 in enumerate(self.method_names):
                        for j, method_2 in enumerate(self.method_names):
                            self.iter += 1
                            statistic, p_value = stats.ttest_ind(data[(method_1, stream)][metric].values, data[(method_2, stream)][metric].values)
                            if p_value < treshold and statistic > 0:
                                ranking[method_1] += 1
                elif self.test_name is "wilcoxon":
                    for i, method_1 in enumerate(self.method_names):
                        for j, method_2 in enumerate(self.method_names):
                            if method_1 == method_2:
                                continue
                            if data[(method_2, stream_name)] is None:
                                ranking[method_1] += 1
                                print("if1", method_2)
                                self.iter += 1
                                continue
                            if data[(method_1, stream_name)] is None:
                                print("if2", method_1)
                                continue

                            self.iter += 1
                            try:
                                statistic, p_value = stats.ranksums(data[(method_1, stream)][metric].values, data[(method_2, stream)][metric].values)
                                if p_value < treshold:
                                    ranking[method_1] += statistic
                            except:
                                print(method_1,method_2)

            if not os.path.exists("results/ranking_tests/%s_%s/" % (self.date_time, stream)):
                os.makedirs("results/ranking_tests/%s_%s/" % (self.date_time, stream))

            trace = self.prepare_trace(ranking)
            stream = stream.split("/")[1]
            layout = go.Layout(title='Ranking %s tests for %s' % (self.test_name, stream), plot_bgcolor='rgb(230, 230, 230)')
            fig = dict(data=[trace], layout=layout)
            plotly.offline.plot(fig, filename="results/ranking_tests/%s_%s/ranking_%s.html" % (self.date_time, stream, self.test_name), auto_open=auto_open)

    def test_metrics(self, treshold=0.001, auto_open=True):
        data = {}
        ranking = {}
        for method_name in self.method_names:
            ranking[method_name] = 0
            for stream_name in self.stream_names:
                try:
                    data[(method_name, stream_name)] = pd.read_csv("results/raw/%s/%s.csv" % (stream_name, method_name), header=0, index_col=0)
                except:
                    print("None is ", method_name, stream_name)
                    data[(method_name, stream_name)] = None

        if not self.metrics:
            self.metrics = data[(self.method_names[0], self.stream_names[0])].columns.values

        for metric in self.metrics:
            ranking = {}
            self.iter = 0
            for method_name in self.method_names:
                ranking[method_name] = 0
            for stream in self.stream_names:
                if self.test_name is "tstudent":
                    for i, method_1 in enumerate(self.method_names):
                        for j, method_2 in enumerate(self.method_names):
                            self.iter += 1
                            statistic, p_value = stats.ttest_ind(data[(method_1, stream)][metric].values, data[(method_2, stream)][metric].values)
                            if p_value < treshold and statistic > 0:
                                ranking[method_1] += 1
                elif self.test_name is "wilcoxon":
                    for i, method_1 in enumerate(self.method_names):
                        for j, method_2 in enumerate(self.method_names):
                            if method_1 == method_2:
                                continue
                            if data[(method_2, stream_name)] is None:
                                print("if1", method_2)
                                continue
                            if data[(method_1, stream_name)] is None:
                                print("if2", method_1)
                                continue

                            try:
                                statistic, p_value = stats.ranksums(data[(method_1, stream)][metric].values, data[(method_2, stream)][metric].values)
                                if p_value < treshold:
                                    ranking[method_1] += statistic
                            except:
                                print(method_1,method_2, stream)

            if not os.path.exists("results/ranking_tests/%s_%s/" % (self.date_time, metric)):
                os.makedirs("results/ranking_tests/%s_%s/" % (self.date_time, metric))

            self.save_to_tex(ranking, metric)
            trace = self.prepare_trace(ranking)
            layout = go.Layout(title='Ranking %s tests for %s' % (self.test_name, metric), plot_bgcolor='rgb(230, 230, 230)')
            fig = dict(data=[trace], layout=layout)
            plotly.offline.plot(fig, filename="results/ranking_tests/%s_%s/ranking_%s.html" % (self.date_time, metric, self.test_name), auto_open=auto_open)

    def save_to_tex(self, ranking, metric):
        items = ranking.items()
        print(ranking)
        with open("results/ranking_tests/%s_%s/ranking_%s_%s.tex" % (self.date_time, metric, metric, self.test_name), "w+") as file:
            vals = []
            names = []
            indexes = range(len(ranking))

            print("\\begin{table}[]", file=file)
            print("\\caption{%s}" % metric.replace("_", " "), file=file)
            print("\\centering", file=file)
            print("\\begin{tabular}{|l|c|l|}", file=file)
            print("\\hline", file=file)
            print("Rank & Algorithm & Ranking \\\\ \\hline", file=file)

            for ind, it in enumerate(sorted(items, key=lambda item: item[1], reverse=True)):
                names.append(it[0])
                vals.append(it[1])
                print("%d & %s & %d \\\\ \\hline" % (ind+1, it[0], it[1]), file=file)

            print("\\end{tabular}", file=file)
            print("\\end{table}", file=file)

    def prepare_trace(self, ranking):
        items = ranking.items()
        print(ranking)

        vals = []
        names = []

        for it in sorted(items, key=lambda item: item[1], reverse=True):
            names.append(it[0])
            vals.append(it[1])
            # vals.append(round(it[1]/float(self.iter)*1000, 2))

        trace = go.Table(
            columnwidth=[20, 50, 50],
            header=dict(values=["<b>Position<b>", "<b>Method<b>", "<b>Score<b>"],
                        line=dict(color='#506784'),
                        fill=dict(color='#119DFF'),
                        align=['center'],
                        font=dict(color='white', size=16),
                        height=32),
            cells=dict(values=[list(range(1, len(names)+1)), names, vals],
                       line=dict(color='#506784'),
                       fill=dict(color=['white']),
                       align=['center'],
                       font=dict(color='#506784', size=16),
                       height=32
                       ))
        return trace

    def pairs_metrics_pie(self, method_names, stream_names, metrics, experiment_name, methods_alias=None, streams_alias=None, metrics_alias=None, treshold=0.10):
        self.method_names = method_names
        self.stream_names = stream_names
        self.metrics = metrics
        self.experiment_name = experiment_name
        self.streams_alias = streams_alias
        if self.metrics_alias is None:
            self.metrics_alias = self.metrics
        if self.methods_alias is None:
            self.methods_alias = self.method_names
        if self.streams_alias is None:
            self.streams_alias = self.stream_names[0].split("/")[0]

        data = {}
        for method_name in self.method_names:
            for stream_name in self.stream_names:
                for metric in self.metrics:
                    try:
                        data[(method_name, stream_name, metric)] = np.genfromtxt("results/raw_metrics/%s/%s/%s/%s.csv" % (self.experiment_name, stream_name, metric, method_name), header=0, index_col=0)
                    except:
                        print("None is ", method_name, stream_name, metric)
                        data[(method_name, stream_name, metric)] = None
                        print(data[(method_name, stream_name, metric)])

        if not self.metrics:
            self.metrics = data[(self.method_names[0], self.stream_names[0])].columns.values

        for metric, metric_a in zip(self.metrics, self.metrics_alias):
            ranking = {}
            self.iter = 0
            for method_name in self.method_names:
                ranking[method_name] = {"win": 0, "lose": 0, "tie": 0}

            for stream in self.stream_names:
                if self.test_name is "tstudent":
                    for i, method_1 in enumerate(self.method_names):
                        for j, method_2 in enumerate(self.method_names):
                            self.iter += 1
                            statistic, p_value = stats.ttest_ind(data[(method_1, stream, metric)], data[(method_2, stream, metric)])
                            if p_value < treshold and statistic > 0:
                                ranking[method_1] += 1
                elif self.test_name is "wilcoxon":
                    for i, method_1 in enumerate(self.method_names):
                        for j, method_2 in enumerate(self.method_names):
                            if method_1 == method_2:
                                continue
                            if data[(method_2, stream, metric)] is None:
                                print("None data", method_2, stream)
                                continue
                            if data[(method_1, stream, metric)] is None:
                                print("None data", method_1, stream)
                                continue

                            try:
                                statistic, p_value = stats.ranksums(data[(method_1, stream, metric)], data[(method_2, stream, metric)])
                                if p_value < treshold:
                                    if statistic > 0:
                                        ranking[method_1]["win"] += 1
                                    else:
                                        ranking[method_1]["lose"] += 1
                                else:
                                    ranking[method_1]["tie"] += 1
                            except:
                                print("Exception", method_1, method_2, stream)

            print(ranking)
            labels = 'Win', 'Lose', 'Tie'
            explode = [0,0,0]
            # mpl.style.use("seaborn")
            # colors = []
            # colors.append(mpl.colors.Colormap("Set1", 0))
            # colors.append(mpl.colors.Colormap("Set1", 2))
            # colors.append(mpl.colors.Colormap("Set1", 5))
            # print(colors)
            my_dpi = 100
            fig, axs = plt.subplots(4,2,figsize=(400/my_dpi, 900/my_dpi), dpi=my_dpi)
            for ax1, method_name, ma in zip(axs.flat, self.method_names, self.methods_alias):
                sizes = ranking[method_name]['win'], ranking[method_name]['lose'], ranking[method_name]['tie']
                ax1.pie(sizes, shadow=False, startangle=90, colors=["green", "crimson", "gold"])
                ax1.axis('equal')
                ax1.set_title(ma)
            fig.legend(labels, loc=8)
            fig.suptitle(metric_a+" "+self.experiment_name+" "+self.streams_alias)
            if not os.path.exists("results/ranking_plots/%s/" % (self.date_time)):
                os.makedirs("results/ranking_plots/%s/" % (self.date_time))
            plt.savefig(fname="results/ranking_plots/%s/%s_%s_%s" % (self.date_time, metric, self.experiment_name, self.streams_alias), dpi=my_dpi)

    def pairs_metrics_old(self, method_names, stream_names, metrics, experiment_name, methods_alias=None, streams_alias=None, metrics_alias=None, treshold=0.10):
        self.method_names = method_names
        self.stream_names = stream_names
        self.metrics = metrics
        self.experiment_name = experiment_name
        self.streams_alias = streams_alias
        self.metrics_alias = metrics_alias
        self.methods_alias = methods_alias
        if self.metrics_alias is None:
            self.metrics_alias = self.metrics
        if self.methods_alias is None:
            self.methods_alias = self.method_names
        if self.streams_alias is None:
            self.streams_alias = self.stream_names[0].split("/")[0]

        data = {}
        for method_name in self.method_names:
            for stream_name in self.stream_names:
                for metric in self.metrics:
                    try:
                        data[(method_name, stream_name, metric)] = np.genfromtxt("results/raw_metrics/%s/%s/%s/%s.csv" % (self.experiment_name, stream_name, metric, method_name))
                    except:
                        print("None is ", method_name, stream_name, metric)
                        data[(method_name, stream_name, metric)] = None
                        print(data[(method_name, stream_name, metric)])

        for metric, metric_a in zip(self.metrics, self.metrics_alias):
            ranking = {}
            self.iter = 0
            for method_name in self.method_names:
                ranking[method_name] = {"win": 0, "lose": 0, "tie": 0}

            for stream in tqdm(self.stream_names, "Rank %s" % metric):
                if self.test_name is "tstudent":
                    for i, method_1 in enumerate(self.method_names):
                        for j, method_2 in enumerate(self.method_names):
                            self.iter += 1
                            statistic, p_value = stats.ttest_ind(data[(method_1, stream, metric)], data[(method_2, stream, metric)])
                            if p_value < treshold and statistic > 0:
                                ranking[method_1] += 1
                elif self.test_name is "wilcoxon":
                    for i, method_1 in enumerate(self.method_names):
                        for j, method_2 in enumerate(self.method_names):
                            if method_1 == method_2:
                                continue
                            if data[(method_2, stream, metric)] is None:
                                print("None data", method_2, stream)
                                continue
                            if data[(method_1, stream, metric)] is None:
                                print("None data", method_1, stream)
                                continue

                            try:
                                statistic, p_value = stats.ranksums(data[(method_1, stream, metric)], data[(method_2, stream, metric)])
                                if p_value < treshold:
                                    if statistic > 0:
                                        ranking[method_1]["win"] += 1
                                    else:
                                        ranking[method_1]["lose"] += 1
                                else:
                                    ranking[method_1]["tie"] += 1
                            except:
                                print("Exception", method_1, method_2, stream, metric)

                rank_win = []
                rank_tie = []
                rank_lose = []
                rank_none = []
                for method_name in self.method_names:
                    rank_win.append(ranking[method_name]['win'])
                    rank_tie.append(ranking[method_name]['tie'])
                    rank_lose.append(ranking[method_name]['lose'])
                    try:
                        rank_none.append(ranking[method_name]['none'])
                    except Exception:
                        pass

                rank_win.reverse()
                rank_tie.reverse()
                rank_lose.reverse()
                rank_none.reverse()

                rank_win = np.array(rank_win)
                rank_tie = np.array(rank_tie)
                rank_lose = np.array(rank_lose)
                rank_none = np.array(rank_none)
                ma = self.methods_alias.copy()
                ma.reverse()
                plt.axes().get_xaxis().set_visible(False)
                plt.rc('ytick', labelsize=30)

                plt.barh(ma, rank_win, color="green", height=0.9)
                plt.barh(ma, rank_tie, left=rank_win, color="gold", height=0.9)
                plt.barh(ma, rank_lose, left=rank_win+rank_tie, color="crimson", height=0.9)
                try:
                    plt.barh(ma, rank_none, left=rank_win+rank_tie+rank_lose, color="black", height=0.9)
                except Exception:
                    pass
                plt.xlim(0, len(self.stream_names)*(len(self.method_names)-1))
                plt.title(metric_a, fontsize=40)

                if not os.path.exists("results/ranking_plots/%s/" % (experiment_name)):
                    os.makedirs("results/ranking_plots/%s/" % (experiment_name))
                plt.gcf().set_size_inches(5, 5)
                plt.savefig(fname="results/ranking_plots/%s/%s_%s_hbar" % (experiment_name, self.streams_alias, metric), bbox_inches='tight')
                plt.clf()

    def pairs_metrics(self, method_names, stream_names, metrics, experiment_name, methods_alias=None, streams_alias=None, metrics_alias=None, treshold=0.5):
        self.method_names = method_names
        self.stream_names = stream_names
        self.metrics = metrics
        self.experiment_name = experiment_name
        self.streams_alias = streams_alias
        self.metrics_alias = metrics_alias
        self.methods_alias = methods_alias
        if self.metrics_alias is None:
            self.metrics_alias = self.metrics
        if self.methods_alias is None:
            self.methods_alias = self.method_names
        if self.streams_alias is None:
            self.streams_alias = self.stream_names[0].split("/")[0]

        data = {}
        for method_name in self.method_names:
            for stream_name in self.stream_names:
                for metric in self.metrics:
                    try:
                        data[(method_name, stream_name, metric)] = np.genfromtxt("results/raw_metrics/%s/%s/%s/%s.csv" % (self.experiment_name, stream_name, metric, method_name))
                    except:
                        print("None is ", method_name, stream_name, metric)
                        data[(method_name, stream_name, metric)] = None
                        print(data[(method_name, stream_name, metric)])

        for metric, metric_a in zip(self.metrics, self.metrics_alias):
            ranking = {}
            self.iter = 0
            for method_name in self.method_names:
                ranking[method_name] = {"win": 0, "lose": 0, "tie": 0}

            for stream in tqdm(self.stream_names, "Rank %s" % metric):
                if self.test_name is "tstudent":
                    for i, method_1 in enumerate(self.method_names):
                        for j, method_2 in enumerate(self.method_names):
                            self.iter += 1
                            statistic, p_value = stats.ttest_ind(data[(method_1, stream, metric)], data[(method_2, stream, metric)])
                            if p_value < treshold and statistic > 0:
                                ranking[method_1] += 1
                elif self.test_name is "wilcoxon":
                    method_1 = self.method_names[0]
                    for j, method_2 in enumerate(self.method_names):
                        if method_1 == method_2:
                            continue
                        if data[(method_2, stream, metric)] is None:
                            print("None data", method_2, stream)
                            continue
                        if data[(method_1, stream, metric)] is None:
                            print("None data", method_1, stream)
                            continue

                        try:
                            statistic, p_value = stats.ranksums(data[(method_1, stream, metric)], data[(method_2, stream, metric)])
                            if p_value < treshold:
                                if statistic > 0:
                                    ranking[method_2]["win"] += 1
                                else:
                                    ranking[method_2]["lose"] += 1
                            else:
                                ranking[method_2]["tie"] += 1
                        except:
                            print("Exception", method_1, method_2, stream, metric)

                rank_win = []
                rank_tie = []
                rank_lose = []
                rank_none = []

                for method_name in self.method_names[1:]:
                    rank_win.append(ranking[method_name]['win'])
                    rank_tie.append(ranking[method_name]['tie'])
                    rank_lose.append(ranking[method_name]['lose'])
                    try:
                        rank_none.append(ranking[method_name]['none'])
                    except Exception:
                        pass

                rank_win.reverse()
                rank_tie.reverse()
                rank_lose.reverse()
                rank_none.reverse()

                rank_win = np.array(rank_win)
                rank_tie = np.array(rank_tie)
                rank_lose = np.array(rank_lose)
                rank_none = np.array(rank_none)
                ma = self.methods_alias[1:].copy()
                ma.reverse()
                # plt.axes().get_xaxis().set_visible(False)
                plt.rc('ytick', labelsize=30)

                plt.barh(ma, rank_win, color="green", height=0.9)
                plt.barh(ma, rank_tie, left=rank_win, color="gold", height=0.9)
                plt.barh(ma, rank_lose, left=rank_win+rank_tie, color="crimson", height=0.9)
                try:
                    plt.barh(ma, rank_none, left=rank_win+rank_tie+rank_lose, color="black", height=0.9)
                except Exception:
                    pass
                plt.xlim(0, len(self.stream_names))
                plt.axvline(20, 0, 1, linestyle="--", linewidth=3, color="black")
                plt.title(metric_a, fontsize=40)

                if not os.path.exists("results/ranking_plots/%s/" % (experiment_name)):
                    os.makedirs("results/ranking_plots/%s/" % (experiment_name))
                plt.gcf().set_size_inches(5, 5)
                plt.savefig(fname="results/ranking_plots/%s/%s_%s_hbar" % (experiment_name, self.streams_alias, metric), bbox_inches='tight')
                plt.clf()


    def pairs_metrics_multi(self, method_names, stream_names, treshold=0.10, experiment_names="", streams_alias=""):
        self.method_names = method_names
        self.stream_names = stream_names
        self.experiment_names = experiment_names
        self.streams_alias = streams_alias

        fig,axs =  plt.subplots(len(self.experiment_names),len(self.metrics))

        for e_index, experiment_name in enumerate(self.experiment_names):
            data = {}
            for method_name in self.method_names:
                for stream_name in self.stream_names:
                    try:
                        data[(method_name, stream_name)] = pd.read_csv("results/raw_%s/%s/%s.csv" % (experiment_name,stream_name, method_name), header=0, index_col=0)
                    except:
                        print("None is ",method_name, stream_name)
                        data[(method_name, stream_name)] = None
                        print(data[(method_name, stream_name)])

            if not self.metrics:
                self.metrics = data[(self.method_names[0], self.stream_names[0])].columns.values

            m_index = -1
            for metric, metric_a in zip(self.metrics, self.metrics_alias):
                m_index += 1
                ranking = {}
                self.iter = 0
                for method_name in self.method_names:
                    ranking[method_name] = {"win": 0, "lose": 0, "tie": 0, "none": 0}

                for stream in self.stream_names:
                    if self.test_name is "tstudent":
                        for i, method_1 in enumerate(self.method_names):
                            for j, method_2 in enumerate(self.method_names):
                                self.iter += 1
                                statistic, p_value = stats.ttest_ind(data[(method_1, stream)][metric].values, data[(method_2, stream)][metric].values)
                                if p_value < treshold and statistic > 0:
                                    ranking[method_1] += 1
                    elif self.test_name is "wilcoxon":
                        for i, method_1 in enumerate(self.method_names):
                            for j, method_2 in enumerate(self.method_names):
                                if method_1 == method_2:
                                    continue
                                if data[(method_2, stream)] is None:
                                    print("None data", method_2, stream)
                                    ranking[method_2]["none"] += 1
                                    continue
                                if data[(method_1, stream)] is None:
                                    print("None data", method_1, stream)
                                    ranking[method_1]["none"] += 1
                                    continue

                                try:
                                    statistic, p_value = stats.ranksums(data[(method_1, stream)][metric].values, data[(method_2, stream)][metric].values)
                                    if p_value < treshold:
                                        if statistic > 0:
                                            ranking[method_1]["win"] += 1
                                        else:
                                            ranking[method_1]["lose"] += 1
                                    else:
                                        ranking[method_1]["tie"] += 1
                                except:
                                    print("Exception", method_1, method_2, stream)

                rank_win = []
                rank_tie = []
                rank_lose = []
                rank_none = []
                for method_name in self.method_names:
                    rank_win.append(ranking[method_name]['win'])
                    rank_tie.append(ranking[method_name]['tie'])
                    rank_lose.append(ranking[method_name]['lose'])
                    rank_none.append(ranking[method_name]['none'])

                rank_win.reverse()
                rank_tie.reverse()
                rank_lose.reverse()
                rank_none.reverse()

                rank_win = np.array(rank_win)
                rank_tie = np.array(rank_tie)
                rank_lose = np.array(rank_lose)
                rank_none = np.array(rank_none)
                ma = self.methods_alias.copy()
                ma.reverse()
                # plt.axes().get_xaxis().set_visible(False)
                # plt.rc('ytick', labelsize=10)

                axs[e_index][m_index].barh(ma, rank_win, color="green", height=0.9)
                axs[e_index][m_index].barh(ma, rank_tie, left=rank_win, color="gold", height=0.9)
                axs[e_index][m_index].barh(ma, rank_lose, left=rank_win+rank_tie, color="crimson", height=0.9)
                axs[e_index][m_index].barh(ma, rank_none, left=rank_win+rank_tie+rank_lose, color="black", height=0.9)
                axs[e_index][m_index].set_xlim([0,len(self.stream_names)*(len(self.method_names)-1)])
                # plt.title(metric_a+" "+experiment_name.upper()+" "+self.streams_alias)

            if not os.path.exists("results/ranking_plots/%s/" % (self.date_time)):
                os.makedirs("results/ranking_plots/%s/" % (self.date_time))
            fig.savefig(fname="results/ranking_plots/%s/%s_%s_%s_hbar)multi" % (self.date_time, self.streams_alias, experiment_name, metric))
            fig.clf()
