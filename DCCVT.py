"""CLI entrypoint wiring for the DCCVT experiment runner."""
# python ./DCCVT.py --args-file ./argfiles/DCCVT_figs_teaser.args 
from dccvt.main import main


if __name__ == "__main__":
    main(__file__)
