import argparse
import sys

#FLAGS.faqPath, FLAGS.csvFolderPath, FLAGS.csvFolderPath,FLAGS.targetLen, FLAGS.reportPath, FLAGS.threshold, FLAGS.format)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()


    parser.add_argument("--trainTsvPath", default="train.tsv",
                        type=str,
                        help="train tsv path")

    parser.add_argument("--testTsvPath", default="test.tsv",
                        type=str,
                        help="test tsv path")

    parser.add_argument("--maxN", default=2,
                        type=int,
                        help="ngrams accepted as features in model")

    parser.add_argument("--split", default=0.2,
                        type=int,
                        help="validation set size")

    parser.add_argument("--seed", default=42,
                        type=int,
                        help="random seed")

    parser.add_argument("--plotFreqSize", default=10,
                        type=int,
                        help="")

    parser.add_argument("--plotWidth", default=10,
                        type=int,
                        help="")

    parser.add_argument("--plotHeight", default=10,
                        type=int,
                        help="")

    parser.add_argument("--predictTsvPath", default="predict.tsv",
                        type=str,
                        help="")








    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)

    print("")
    args = parser.parse_args()
    for arg in vars(args):
        print("{}={}".format(arg.upper(), getattr(args, arg)))
    print("")

    return args


FLAGS = parse_args()

#FAQSIZE=100
#SIMFUNC=lev
#0.27175045013427734

#FAQSIZE=1000
#SIMFUNC=lev
#0.3051886558532715

#FAQSIZE=10000
#SIMFUNC=lev
#0.45138001441955566

#FAQSIZE=99999
#SIMFUNC=lev
#0.5071938037872314

#FAQSIZE=99999
#SIMFUNC=cos
#97.42133283615112

#FAQSIZE=10000
#SIMFUNC=cos
#76.32733488082886

#FAQSIZE=1000
#SIMFUNC=cos
#7.74871826171875

#FAQSIZE=100
#SIMFUNC=cos
#0.9931180477142334

#speech recognition result from ortz
#FAQSIZE=99999
#SIMFUNC=cos
#96.3589038848877
#correct
