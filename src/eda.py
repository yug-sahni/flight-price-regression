"""Exploratory data analysis: prints key statistics and saves figures to outputs/figures."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIG, OUT, seed_everything
from .features import load_raw, build_features

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
C = {"Economy": "#2a7ab0", "Business": "#d9622b"}


def main():
    seed_everything()
    tr, te = load_raw()
    stats = {}
    stats["n_train"], stats["n_test"] = len(tr), len(te)
    stats["missing_pct_train"] = (tr.isna().mean() * 100).round(2)[lambda s: s > 0].to_dict()
    stats["missing_pct_test"] = (te.isna().mean() * 100).round(2)[lambda s: s > 0].to_dict()
    stats["price_skew_raw"] = float(tr.price.skew())
    stats["price_skew_log"] = float(np.log1p(tr.price).skew())
    stats["median_price_by_class"] = tr.groupby("flight_class").price.median().to_dict()
    stats["corr_duration_price"] = float(tr[["duration", "price"]].corr().iloc[0, 1])
    stats["corr_daysleft_price"] = float(tr[["days_left", "price"]].corr().iloc[0, 1])
    stats["duplicate_feature_rows"] = int(tr.duplicated(subset=[c for c in tr.columns if c not in ("id", "price")]).sum())
    stats["unseen_flights_in_test_pct"] = float((~te.flight.isin(tr.flight)).mean() * 100)

    # 1. target distribution
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].hist(tr.price, bins=60, color="#2a7ab0"); ax[0].set_title(f"price (skew {stats['price_skew_raw']:.2f})")
    ax[1].hist(np.log1p(tr.price), bins=60, color="#2a7ab0"); ax[1].set_title(f"log1p(price) (skew {stats['price_skew_log']:.2f})")
    fig.tight_layout(); fig.savefig(FIG / "01_target_distribution.png"); plt.close(fig)

    # 2. missing values
    miss = pd.DataFrame({"train": tr.isna().mean() * 100, "test": te.isna().mean() * 100}).drop(index="price")
    miss = miss[(miss > 0).any(axis=1)]
    ax = miss.plot.bar(figsize=(5, 3.5), color=["#2a7ab0", "#d9622b"]); ax.set_ylabel("% missing"); ax.set_title("Missing values")
    plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(FIG / "02_missing.png"); plt.close()

    # 3. price by class & stops
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    tr.boxplot(column="price", by="flight_class", ax=ax[0], grid=False); ax[0].set_title("price by class"); ax[0].set_xlabel("")
    tr.boxplot(column="price", by=["flight_class", "stops"], ax=ax[1], grid=False, rot=30); ax[1].set_title("price by class x stops"); ax[1].set_xlabel("")
    fig.suptitle(""); fig.tight_layout(); fig.savefig(FIG / "03_price_class_stops.png"); plt.close(fig)

    # 4. duration vs price
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, g in tr.groupby("flight_class"):
        ax.scatter(g.duration, g.price, s=4, alpha=.35, c=C[k], label=k)
    ax.set_xlabel("duration (h)"); ax.set_ylabel("price"); ax.legend(markerscale=3); ax.set_title("Duration vs price")
    fig.tight_layout(); fig.savefig(FIG / "04_duration_price.png"); plt.close(fig)

    # 5. days_left vs median price
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for k, g in tr.groupby("flight_class"):
        s = g.groupby("days_left").price.median(); ax.plot(s.index, s.values, c=C[k], label=k)
    ax.set_xlabel("days left"); ax.set_ylabel("median price"); ax.legend(); ax.set_title("Booking lead time vs median price")
    fig.tight_layout(); fig.savefig(FIG / "05_daysleft_price.png"); plt.close(fig)

    # 6. airline median price
    s = tr.assign(airline=tr.airline.fillna("(missing)")).groupby("airline").price.median().sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.5)); s.plot.barh(ax=ax, color="#2a7ab0"); ax.set_xlabel("median price"); ax.set_title("Median price by airline")
    fig.tight_layout(); fig.savefig(FIG / "06_airline_price.png"); plt.close(fig)

    # 7. route heatmap
    p = tr.groupby(["source_city", "destination_city"]).price.median().unstack()
    fig, ax = plt.subplots(figsize=(6, 4.5)); im = ax.imshow(p.values, cmap="viridis")
    ax.set_xticks(range(len(p.columns))); ax.set_xticklabels(p.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(p.index))); ax.set_yticklabels(p.index); fig.colorbar(im, label="median price"); ax.set_title("Median price by route")
    fig.tight_layout(); fig.savefig(FIG / "07_route_heatmap.png"); plt.close(fig)

    # 8. correlation of engineered numeric features
    X, y, _, _ = build_features(tr, te)
    num = X[["duration", "days_left", "stops_n", "is_business", "dep_ord", "arr_ord", "flight_num"]].assign(price=y, log_price=np.log1p(y))
    cm = num.corr(); fig, ax = plt.subplots(figsize=(6, 5)); im = ax.imshow(cm, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=60, ha="right"); ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.columns)
    fig.colorbar(im); ax.set_title("Correlation (numeric features)"); fig.tight_layout(); fig.savefig(FIG / "08_correlation.png"); plt.close(fig)

    stats["corr_with_price"] = cm["price"].round(3).to_dict()
    # within-flight price repeatability: how much of log-price variance is explained by flight+class identity alone
    lp = np.log1p(tr.price); key = tr.flight + "|" + tr.flight_class
    stats["share_logprice_var_explained_by_flight_class"] = float(1 - (lp - lp.groupby(key).transform("mean")).var() / lp.var())
    stats["flights_with_single_row"] = int((key.map(key.value_counts()) == 1).sum())
    (OUT / "eda_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
