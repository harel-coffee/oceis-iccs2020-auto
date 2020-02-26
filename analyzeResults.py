from utils import calculate_metrics
from utils import Ploting, Ranking

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
streams = []

# directory = "gen/incremental/"
# mypath = "streams/%s" % directory
# streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
#
# directory = "gen/sudden/"
# mypath = "streams/%s" % directory
# streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

                                                            # step init
streams += ["real/abalone-17_vs_7-8-9-10"]                  # 250 500
streams += ["real/elecNormNew"]                             # 500 1000
streams += ["real/jm1"]                                     # 100 200
streams += ["real/kc1"]                                     # 100 200
streams += ["real/kc2"]                                     # 50 100
streams += ["real/kr-vs-k-three_vs_eleven"]                 # 250 500
streams += ["real/kr-vs-k-zero-one_vs_draw"]                # 250 500
streams += ["real/page-blocks0"]                            # 500 1000
streams += ["real/segment0"]                                # 100 200
streams += ["real/shuttle-1vs4"]                            # 100 200
streams += ["real/vehicle0"]                                # 100 200
streams += ["real/yeast1"]                                  # 150 300
streams += ["real/yeast3"]                                  # 150 300
streams += ["real/wisconsin"]                               # 50 100
streams += ["real/australian"]                              # 50 100
streams += ["real/pima"]                                    # 50 100
streams += ["real/heart"]                                   # 52 52
streams += ["real/glass0"]                                  # 30 60
streams += ["real/glass-0-1-2-3_vs_4-5-6"]                  # 30 60
streams += ["real/glass1"]                                  # 30 60
streams += ["real/yeast-0-2-5-7-9_vs_3-6-8"]                # 100 200
streams += ["real/vowel0"]                                  # 100 200
streams += ["real/yeast-0-2-5-6_vs_3-7-8-9"]                # 100 200
streams += ["real/yeast-0-3-5-9_vs_7-8"]                    # 70 140
streams += ["real/yeast-2_vs_4"]                            # 60 120
streams += ["real/yeast-0-5-6-7-9_vs_4"]                    # 60 120
streams += ["real/shuttle-5vsA"]                            # 1000 2000
streams += ["real/shuttle-1vsA"]                            # 1000 2000
streams += ["real/shuttle-4-5vsA"]                          # 1000 2000
streams += ["real/shuttle-4vsA"]                            # 1000 2000

method_names = [
                "OCEIS",
                "KMeanClustering",
                "LearnppCDS",
                "LearnppNIE",
                "REA",
                "OUSE",
                "MLPClassifier",
                ]

methods_alias = [
                "OCEIS",
                "KMC",
                "L++CDS",
                "L++NIE",
                "REA",
                "OUSE",
                "MLPC",
                ]

metrics_alias = [
           "Gmean",
           "Precision",
           "Recall",
           "Specifity",
          ]

metrics = [
           "g_mean",
           "precision",
           "recall",
           "specifity",
          ]

experiment_name = "final"

calculate_metrics(method_names, streams, metrics, experiment_name, recount=True)

plt = Ploting()
plt.plot_streams_matplotlib(method_names, streams, metrics, experiment_name, gauss=3, methods_alias=methods_alias, metrics_alias=metrics_alias)

rnk = Ranking()
rnk.pairs_metrics(method_names, streams, metrics, experiment_name, methods_alias=methods_alias, metrics_alias=metrics_alias)
