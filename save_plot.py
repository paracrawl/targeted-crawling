from tldextract import extract
import matplotlib
matplotlib.use('Agg')
import pylab as plt

from other_strategies import dumb, randomCrawl, balanced, byCrawlDate, linkText

######################################################################################
def SavePlot(params, env, createFigure, saveDirPlots, epoch, sset, arrRL, totReward, totDiscountedReward):
    url = env.rootURL
    domain = extract(url).domain

    if createFigure:
        crawlLen = min(params.maxCrawl, len(env.nodes))
        arrBreadth = dumb(env, crawlLen, params, 0)
        arrDepth = dumb(env, crawlLen, params, 1)
        arrRandom = randomCrawl(env, crawlLen, params)
        arrBalanced = balanced(env, crawlLen, params)
        arrLinkText = linkText(env, crawlLen, params)
        arrByCrawlDate = byCrawlDate(env, crawlLen, params)
        #print("arrRL", len(arrRL))
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(arrBreadth, label="breadth-first {0:.1f}".format(arrBreadth[-1]), color='black')
        ax.plot(arrDepth, label="depth-first {0:.1f}".format(arrDepth[-1]), color='green')
        ax.plot(arrRandom, label="random {0:.1f}".format(arrRandom[-1]), color='firebrick')
        ax.plot(arrBalanced, label="balanced {0:.1f}".format(arrBalanced[-1]), color='red')
        ax.plot(arrLinkText, label="link text {0:.1f}".format(arrLinkText[-1]), color='yellow')
        ax.plot(arrByCrawlDate, label="crawl date {0:.1f}".format(arrByCrawlDate[-1]), color='blue')
        ax.plot(arrRL, label="RL {0:.1f} {1:.1f} {2:.1f}".format(arrRL[-1], totReward, totDiscountedReward), color='salmon')

        ax.legend(loc='upper left')
        plt.xlabel('#crawled')
        plt.ylabel('#found')
        plt.title("{sset} {domain}".format(sset=sset, domain=domain))

        fig.savefig("{dir}/{sset}-{domain}-{epoch}.png".format(dir=saveDirPlots, sset=sset, domain=domain, epoch=epoch))
        plt.close()

    # output scores
    f = open("{dir}/{domain}-rewards.txt".format(dir=saveDirPlots, domain=domain), "a")
    out = "\t".join([str(epoch), str(arrRL[-1]), str(totReward), str(totDiscountedReward)])
    f.write(out + "\n")
    f.close()
    