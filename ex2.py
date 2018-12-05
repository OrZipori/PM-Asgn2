# Or Zipori 302933833
# Shauli Ravfogel 308046861
import sys
import utils

def main():
    data = utils.fetchFile('./dataset/develop.txt')
    ho = utils.Headout(data)
    lidstone = utils.Lidstone(data, lambdaP=0.5)

    ho.train()
    lidstone.train()

    print("Debug HO", utils.debugModel(ho))
    print("Debug Lidstone", utils.debugModel(lidstone))
    print("Validation Lidstone (perplexity)", lidstone.validate())

if __name__ == "__main__":
    main()