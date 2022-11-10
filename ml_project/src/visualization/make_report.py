import pandas as pd
import sweetviz as sv


if __name__ == "__main__":
    df = pd.read_csv("data/raw/heart_cleveland_upload.csv")
    report = sv.analyze(df)
    report.show_html("reports/report.html", open_browser=False)
