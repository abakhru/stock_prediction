template = """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Title of the document</title>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/milligram/1.3.0/milligram.min.css">
  <style>
    .tradingview-widget-container {{
      position: sticky;
      top: 20px;
    }}
    .stocks-view {{
      display: flex;
      flex-wrap: nowrap;
    }}
    .stocks-listing {{
      width: 780px;
      flex-wrap: nowrap;
      padding: 20px;
    }}
    .stocks-graph {{
      flex-wrap: nowrap;
      padding: 20px;
    }}
    th.sticky-header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background-color: white;
    }}
    .positive-movement {{
      color: green;
      font-weight: bold;
    }}
    .negative-movement {{
      color: red;
      font-weight: bold;
    }}
    .blue-category {{
      background-color: lightsteelblue;
    }}
  </style>
</head>

<body>
{}
<div class="stocks-view">
  <div class="stocks-listing">
    <table>
      <thead>
        <tr>
          <th class="sticky-header">Symbol</th>
          <th class="sticky-header">April 1 2019</th>
          <th class="sticky-header">Dec 2 2019</th>
          <th class="sticky-header">Today</th>
          <th class="sticky-header">Movement since April 1 2019</th>
          <th class="sticky-header">Movement since Dec 2 2019</th>
          <th class="sticky-header">Bankruptcy probability</th>
        </tr>
      </thead>
      <tbody>
        {}
      </tbody>
    </table>

  </div>
  <div class="stocks-graph"
  <!-- TradingView Widget BEGIN -->
  <div class="tradingview-widget-container">
    <div id="tradingview_63a66"></div>
    <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/symbols/AAPL/" rel="noopener" target="_blank"><span class="blue-text">AAPL Chart</span></a> by TradingView</div>
  </div>
  <!-- TradingView Widget END -->
  </div>
</div>

<script type="text/javascript">
  function renderChart(symbol) {{
    new TradingView.widget(
    {{
      "width": 750,
      "height": 500,
      "symbol": symbol,
      "interval": "180",
      "timezone": "Etc/UTC",
      "theme": "light",
      "style": "1",
      "locale": "en",
      "toolbar_bg": "#f1f3f6",
      "enable_publishing": false,
      "allow_symbol_change": true,
      "container_id": "tradingview_63a66"
    }}
    );
  }}

  document.addEventListener('DOMContentLoaded', function(){{ 
    renderChart('BA');
  }}, false);
</script>
</body>

</html>"""
