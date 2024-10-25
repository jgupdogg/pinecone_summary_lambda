import datetime

def format_date(date_obj):
    return date_obj.strftime('%Y-%m-%d')

def generate_date_range(start_date, end_date):
    dates = []
    delta = datetime.timedelta(days=1)
    while start_date <= end_date:
        dates.append(format_date(start_date))
        start_date += delta
    return {"$in": dates}

def generate_recent_dates(days=5):
    dates = []
    for i in range(days):
        date_obj = datetime.datetime.now() - datetime.timedelta(days=i)
        dates.append(format_date(date_obj))
    return {"$in": dates}
