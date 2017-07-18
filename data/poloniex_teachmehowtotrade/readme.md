source: http://teachmehowto.trade/PoloData/#section-ohlcv
post:https://www.reddit.com/r/BitcoinMarkets/comments/694q0a/historical_pricing_data_for_poloniex_btceth_pairs

----

Hey guys -

By popular demand, I have made historical up-to-date OHLCV (open-high-low-close-volume) data for the most-popular cryptocurrency pairs traded on Poloniex available for download.

**Link: http://teachmehowto.trade/PoloData/**

The format looks like:

          date       high        low       open      close   volume quoteVolume weightedAverage
    1493848800 0.05337888 0.05304304 0.05334496 0.05309167 203.0503    3819.472      0.05316189
    1493850600 0.05319993 0.05271523 0.05309168 0.05299500 319.6935    6033.497      0.05298644
    1493852400 0.05372393 0.05297500 0.05299500 0.05357000 572.1481   10695.544      0.05349406
    1493854200 0.05366399 0.05267554 0.05357000 0.05311562 567.2399   10675.820      0.05313314
    1493856000 0.05313172 0.05270000 0.05311563 0.05288821 367.8146    6943.378      0.05297344
    1493857800 0.05310932 0.05286593 0.05300000 0.05303735 159.6984    3011.262      0.05303369

Where `date` is the unix timestamp in UTC.

-------------------------

**Frequently Asked Questions**

> How often is the raw data updated?

The .csv files are updated every hour (for now).

> Why is the data only every 30-minutes?

Based on my experience, 30-minute OHLCV data is the minimum periodicity for any sort of meaningful analysis. Can you build models/strategies/algos to trade 5-minute (or less) intervals? Certainly, but this won't be the best way to get at that data. Additionally, aggregating from 30-minute to 1H/2H/4H/6H/12H/24H intervals is trivial.

> What if I *really* want less-than-30-minute time-series data? Can I get that somehow?

Yes, you can. I built a package for `R`, which wraps a few of the public methods for Poloniex's API. You can use that to get 5- or 15-minute data. Check it out here: **https://github.com/Rb-Engineering/PoloniexR**

> What about order book data?

I'm currently finalizing the back-end for that and it will be available soon! If you can't wait, and you're comfortable working in `R` (or want to learn), check out the `getOrderBook()` function in my `PoloniexR` package above.

-------------------------

If you guys have any questions/comments/concerns, feel free to comment below or PM me.

One last friendly reminder... please **DO NOT ABUSE THIS!!** (e.g. by repeatedly downloading the same file needlessly) This is my gift to the community so all I ask is that you be mindful of the fact that I built this and am hosting it... for free. :)

Happy trading!!
