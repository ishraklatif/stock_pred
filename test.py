# from openbb import obb

# df = obb.equity.price.historical("DSEX", provider="yfinance").to_dataframe()
# print(df.head())
import inspect
import bdshare

funcs = [name for name, obj in inspect.getmembers(bdshare, inspect.isfunction)]
print("Top-level functions in bdshare:")
for f in funcs:
    print("-", f)



