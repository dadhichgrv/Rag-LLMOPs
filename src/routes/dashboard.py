# from fastapi import APIRouter
# from src.app.ingestion.azure_storage import get_blob_data

# router = APIRouter()

# @router.get("/metrics")
# def get_metrics():
#     return get_blob_data()


from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from src.app.ingestion.azure_storage import get_blob_data # the function from the previous script

router = APIRouter()
templates = Jinja2Templates(directory="templates")  # adjust to wherever your .html files live


def flatten_metrics(raw_data: list[dict]) -> list[dict]:
    """Flatten get_all_blob_data()'s nested shape into flat rows the
    template can read directly, and rename fields to match what the
    template actually checks for (e.g. total_asset -> total_assets).
    """
    rows = []
    for doc in raw_data:
        metrics = doc.get("metrics", {})
        report_company = metrics.get("company") or doc.get("company") or "Unknown"
        report_year = metrics.get("year") or doc.get("year")
        report_key = doc.get("blob_name") or "::".join(
            (str(report_company), str(report_year or "unknown"))
        )
        rows.append({
            "report_key": report_key,
            "company": report_company,
            "year": report_year,
            "revenue": metrics.get("revenue"),
            "net_income": metrics.get("net_income") or metrics.get("profit"),
            "operating_income": metrics.get("operating_income"),
            "cash_flow": metrics.get("cash_flow"),
            "total_assets": metrics.get("total_assets") or metrics.get("total_asset"),
            "total_liabilities": metrics.get("total_liabilities"),
            "risk_factors": metrics.get("risk_factors"),
            "growth_drivers": metrics.get("growth_drivers"),
        })
    return rows


@router.get("/dashboard")
def dashboard(request: Request):
    raw_data = get_blob_data()
    metrics = flatten_metrics(raw_data)

    total_companies = len({row["company"] for row in metrics if row.get("company")})
    total_reports = len(metrics)
 
    # Must return a rendered template, not the raw data --
    # `return metrics` skips the HTML entirely and just serializes JSON.
    return templates.TemplateResponse(
        "templates/dashboard.html",
        {
            "request": request,
            "metrics": metrics,
            "total_companies": total_companies,
            "total_reports": total_reports,
        },
    )

    # return templates.TemplateResponse(
    #     # "templates/dashboard.html",  # the file containing your kpi-card HTML
    #     {"request": request, "metrics": metrics}
    # )
    