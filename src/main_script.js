page_origin =
  "*";
weights_url =
  "https://storage.googleapis.com/eecs6893_project/apache_data/weights_ML2_GBReg_df.csv";
return_url =
  "https://storage.googleapis.com/eecs6893_project/dummy_data/return.csv";
sharpe_url =
  "https://storage.googleapis.com/eecs6893_project/dummy_data/sharpe.csv";

function createPie(svgId, data, domain, date) {
  // Set the dimensions and margins of the pie graph
  var margin_pie = { top: 10, right: 30, bottom: 30, left: 30 },
    width_pie = 850 - margin_pie.left - margin_pie.right,
    height_pie = 850 - margin_pie.top - margin_pie.bottom;
  radius = Math.min(width_pie, height_pie) / 2;
  // Pie graph svg setup
  var svg_pie = d3.select(svgId);
  svg_pie = svg_pie.html(null);
  svg_pie = svg_pie
    .append("svg")
    .attr("width", width_pie)
    .attr("height", height_pie)
    .append("g")
    .attr(
      "transform",
      "translate(" + width_pie / 2 + "," + height_pie / 2 + ")"
    );

  // Pie layout
  const path = d3
    .arc()
    .outerRadius(radius - 10)
    .innerRadius(0);
  const labelArc = d3
    .arc()
    .outerRadius(radius - 40)
    .innerRadius(radius - 40);
  const pie = d3.pie().value(function (d) {
    return d.percent;
  });

  // Pie graph color scale
  const color = d3.scaleOrdinal().domain(domain).range(d3.schemeCategory10);
  // Create pie graph
  const g = svg_pie
    .selectAll(".arc")
    .data(pie(data))
    .enter()
    .append("g")
    .attr("class", "arc");

  g.append("path")
    .attr("d", path)
    .style("fill", (d) => color(d.data.ticker));

  g.append("text")
    .attr("transform", (d) => "translate(" + labelArc.centroid(d) + ")")
    .attr("dy", ".35em")
    .text((d) => (d.data.percent != 0 ? d.data.ticker + "%" : ""))
    .style("text-anchor", "middle")
    .style("font-size", 14);

  g.append("text")
    .attr("transform", (d) => "translate(" + labelArc.centroid(d) + ")")
    .attr("dy", "1.7em")
    .text((d) => (d.data.percent != 0 ? d.data.percent + "%" : ""))
    .style("text-anchor", "middle")
    .style("font-size", 12);
}

function createLineGraph(
  svgId,
  data,
  value_column,
  min_value,
  max_value,
  xlabel,
  ylabel,
  xaxisloc
) {
  // Set the dimesions and margins of the histogram graph
  var margin = { top: 30, right: 30, bottom: 30, left: 50 },
    width = 660 - margin.left - margin.right,
    height = 600 - margin.top - margin.bottom;

  const svg = d3
    .select(svgId)
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .style("margin", "30px 0px")
    .append("g")
    .attr("transform", `translate(${margin.left}, ${margin.top})`);

  const grouped_data = d3.group(data, (d) => d.Type);

  const xScale = d3
    .scaleTime()
    .domain(d3.extent(data, (d) => d["Date"]))
    .range([0, width - 180]);

  const yScale = d3
    .scaleLinear()
    .domain([min_value, max_value])
    .range([height - 10, 10]);

  svg
    .append("g")
    .attr("transform", `translate(0, ${yScale(xaxisloc)})`)
    .call(d3.axisBottom(xScale).tickFormat(d3.timeFormat("%m-%Y")));

  svg.append("g").call(d3.axisLeft(yScale));

  const color = d3
    .scaleOrdinal()
    .domain(grouped_data.keys())
    .range(d3.schemeSet1);

  const line = d3
    .line()
    .x((d) => {
      return xScale(d["Date"]);
    })
    .y((d) => yScale(d[value_column]));

  svg
    .selectAll(".line")
    .data(grouped_data)
    .enter()
    .append("path")
    .attr("fill", "none")
    .attr("stroke", (d) => color(d[0]))
    .attr("stroke-width", 1.5)
    .attr("d", (d) => line(Array.from(d.values())[1]));

  // Add X axis label:
  svg
    .append("text")
    .attr("text-anchor", "middle")
    .attr("x", width / 2 - margin.right * 2)
    .attr("y", height + margin.top)
    .text(xlabel);
  // Y axis label:
  svg
    .append("text")
    .attr("text-anchor", "middle")
    .attr("y", 0)
    .attr("x", 0)
    .text(ylabel);

  var legend = svg
    .selectAll(".g")
    .data(grouped_data)
    .enter()
    .append("g")
    .attr("class", "legend");

  legend
    .append("rect")
    .attr("x", width - 162)
    .attr("y", function (d, i) {
      return i * 30;
    })
    .attr("width", 10)
    .attr("height", 10)
    .style("fill", function (d) {
      return color(d[0]);
    });

  legend
    .append("text")
    .attr("x", width - 145)
    .attr("y", function (d, i) {
      return i * 30 + 9;
    })
    .text(function (d) {
      return d[0];
    });
}

function createStackedLineGraph(svgId, data) {
  var margin = { top: 30, right: 50, bottom: 30, left: 20 },
    width = 700 - margin.left - margin.right,
    height = 600 - margin.top - margin.bottom;

  const svg = d3
    .select(svgId)
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", `translate(${margin.left}, ${margin.top})`);
  // Determine the series that need to be stacked.
  const series = d3
    .stack()
    .keys(d3.union(data.map((d) => d.ticker))) // distinct series keys, in input order
    .value(([, D], key) => D.get(key).weight)(
    // get value for each series key and stack
    d3.index(
      data,
      (d) => d.data_date,
      (d) => d.ticker
    )
  ); // group by stack then series key

  // Prepare the scales for positional and color encodings.
  const x = d3
    .scaleUtc()
    .domain(d3.extent(data, (d) => d.data_date))
    .range([margin.left, width - margin.right]);

  const y = d3
    .scaleLinear()
    .domain([0, d3.max(series, (d) => d3.max(d, (d) => d[1]))])
    .rangeRound([height - margin.bottom, margin.top]);

  const color = d3
    .scaleOrdinal()
    .domain(series.map((d) => d.key))
    .range(d3.schemeTableau10);

  // Construct an area shape.
  const area = d3
    .area()
    .x((d) => x(d.data[0]))
    .y0((d) => y(d[0]))
    .y1((d) => y(d[1]));

  // Add the y-axis, remove the domain line, add grid lines and a label.
  svg
    .append("g")
    .attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(height / 80))
    .call((g) => g.select(".domain").remove())
    .call((g) =>
      g
        .selectAll(".tick line")
        .clone()
        .attr("x2", width - margin.left - margin.right)
        .attr("stroke-opacity", 0.1)
    )
    .call((g) =>
      g
        .append("text")
        .attr("x", -margin.left)
        .attr("y", 10)
        .attr("fill", "currentColor")
        .attr("text-anchor", "start")
        .text("Weights(%)")
    );

  // Append a path for each series.
  svg
    .append("g")
    .selectAll()
    .data(series)
    .join("path")
    .attr("fill", (d) => color(d.key))
    .attr("d", area)
    .append("title")
    .text((d) => d.key);

  // Append the horizontal axis atop the area.
  svg
    .append("g")
    .attr("transform", `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).tickFormat(d3.timeFormat("%m-%Y")).tickSizeOuter(0));
  svg
    .append("text")
    .attr("text-anchor", "middle")
    .attr("x", width / 2)
    .attr("y", height + margin.top / 4)
    .style("font-size", "12px")
    .text("Date");

  var legend = svg
    .selectAll(".g")
    .data(series)
    .enter()
    .append("g")
    .attr("class", "legend");

  legend
    .append("rect")
    .attr("x", width-margin.right+10)
    .attr("y", function (d, i) {
      return i * 30;
    })
    .attr("width", 10)
    .attr("height", 10)
    .style("fill", function (d) {
      return color(d.key);
    });

  legend
    .append("text")
    .attr("x", width-margin.right+30)
    .attr("y", function (d, i) {
      return i * 30 + 9;
    })
    .text(function (d) {
      return d.key;
    });
}

// Create pie chart of recommended weights
d3.csv(weights_url, { mode: "cors", headers: { origin: page_origin } }).then(
  function (data) {
    // Holds the weights for all the days
    weights = new Map();
    all_dates = [];
    // Holds the balances for all the days
    balances = [];
    // Holds stacked data for stacked line graph
    stacked_data = [];
    // date_key = "DATE";
    // multiply_percent = false;
    // format_date = false;
    date_key = "";
    multiply_percent = true;
    format_date = true;
    parseDate = d3.timeParse("%m/%d/%y");
    data.forEach(function (d) {
      // Parse the date and weights for each day in the csv
      date = "";
      day_weights = [];
      tickers = [];
      balance = 0;
      for ([key, val] of Object.entries(d)) {
        if (key == date_key) {
          if (format_date) {
            date_parts = val.split("-");
            date = date_parts[1] + "-" + date_parts[2] + "-" + date_parts[0];
          } else {
            date = val;
          }
        } else {
          if (key != "CurrentBalance" && key != "Balance") {
            weight = parseFloat(val.slice(0, -1));
            weight = multiply_percent ? (weight * 100).toFixed(2) : weight;
            day_weights.push({ percent: weight, ticker: key });
            tickers.push(key);
          } else if (key == "Balance" || key == "Current Balance") {
            balance = parseFloat(val.replace(/,/g, "")).toFixed(2);
          }
        }
      }
      weights.set(date, { weights: day_weights, domain: tickers });
      all_dates.push(date);
      balances.push({
        Date: parseDate(date),
        Type: "Portfolio Value ($)",
        Value: balance,
      });
      day_weights.forEach(function (d) {
        stacked_data.push({
          data_date: parseDate(date),
          ticker: d.ticker,
          weight: d.percent,
        });
      });
    });

    all_dates = all_dates.reverse();
    display_dates = Array.from(all_dates);
    display_dates[0] = display_dates[0] + " (Latest Trading Day)";

    // A function that update the pie chart
    function update(selectedDate) {
      current_weights = weights.get(selectedDate);

      // Create pie chart for selected trading day
      createPie(
        "#weightsPie",
        current_weights.weights,
        current_weights.domain,
        selectedDate
      );
    }

    // add the options to the button
    d3.select("#selectDateDropdown")
      .selectAll("myOptions")
      .data(display_dates)
      .enter()
      .append("option")
      .text(function (d) {
        return d;
      }) // text shown in the menu
      .attr("value", function (d) {
        return d.split(" ")[0];
      }); // corresponding value returned

    d3.select("#selectDateDropdown").on("change", function (d) {
      // recover the date that has been chosen
      var selectedDate = d3.select(this).property("value");
      // run the update function with this selected date
      update(selectedDate);
    });

    last_date = all_dates[0];
    current_date = last_date;
    update(last_date);
    // Create line graph for portfolio value
    createLineGraph(
      "#valueChart",
      balances,
      "Value",
      850000,
      1100000,
      "Date",
      "Value ($)",
      850000
    );

    // Create stacked line graph for weight suggestions over time
    createStackedLineGraph("#stackedWeightsGraph", stacked_data);

    last_balance = balances[balances.length-1]
    current_balance_text = 'Last Balance Ending ' + d3.timeFormat('%m-%d-%Y')(last_balance.Date) + ':';
    current_balance_value = '$' + last_balance.Value;
    d3.select("body").select('#currentValueLabel').text(current_balance_text);
    d3.select("body").select('#currentValueText').text(current_balance_value);
  }
);

d3.csv(return_url, { mode: "cors", headers: { origin: page_origin } }).then(
  function (data) {
    date_column = "Date";
    benchmark_column = "AverageReturn(S&P500)";
    portfolio_column = "AverageReturn(Portfolio)";
    parseDate = d3.timeParse("%b %Y");
    processed_data = [];
    min_value = 0;
    max_value = 0;
    data.forEach(function (d) {
      d[benchmark_column] = parseFloat(
        d[benchmark_column].slice(0, -1)
      ).toFixed(2);
      d[portfolio_column] = parseFloat(
        d[portfolio_column].slice(0, -1)
      ).toFixed(2);
      formatted_data_bm = {
        Date: parseDate(d[date_column]),
        Type: benchmark_column,
        Return: d[benchmark_column],
      };
      formatted_data_pf = {
        Date: parseDate(d[date_column]),
        Type: portfolio_column,
        Return: d[portfolio_column],
      };
      processed_data.push(formatted_data_bm);
      processed_data.push(formatted_data_pf);
      min_value = Math.min(min_value, d[benchmark_column], d[portfolio_column]);
      max_value = Math.max(max_value, d[benchmark_column], d[portfolio_column]);
    });
    createLineGraph(
      "#returnsChart",
      processed_data,
      "Return",
      min_value,
      max_value,
      "Date",
      "Return (%)",
      0
    );
  }
);

d3.csv(sharpe_url, { mode: "cors", headers: { origin: page_origin } }).then(
  function (data) {
    date_column = "Date";
    benchmark_column = " Sharpe ratio (S&P 500)";
    portfolio_column = " Sharpe ratio (Portfolio)";
    parseDate = d3.timeParse("%b %Y");
    processed_data = [];
    min_value = 0;
    max_value = 0;
    data.forEach(function (d) {
      d[benchmark_column] = parseFloat(d[benchmark_column].trim()).toFixed(2);
      d[portfolio_column] = parseFloat(d[portfolio_column].trim()).toFixed(2);
      formatted_data_bm = {
        Date: parseDate(d[date_column]),
        Type: benchmark_column,
        Sharpe: d[benchmark_column],
      };
      formatted_data_pf = {
        Date: parseDate(d[date_column]),
        Type: portfolio_column,
        Sharpe: d[portfolio_column],
      };
      processed_data.push(formatted_data_bm);
      processed_data.push(formatted_data_pf);
      min_value = Math.min(min_value, d[benchmark_column], d[portfolio_column]);
      max_value = Math.max(max_value, d[benchmark_column], d[portfolio_column]);
    });

    last_sharpe = data[data.length-1];

    portfolio_text =  portfolio_column.trim()+' Ending ' + last_sharpe[date_column] + ':';
    portfolio_value = last_sharpe[portfolio_column];
    d3.select("body").select('#currentSharpe').text(portfolio_text);
    d3.select("body").select('#currentSharpeValueText').text(portfolio_value);

    bm_text =  benchmark_column.trim()+' Ending ' + last_sharpe[date_column] + ':';
    bm_value = last_sharpe[benchmark_column];
    d3.select("body").select('#currentSharpeBm').text(bm_text);
    d3.select("body").select('#currentSharpeValueTextBm').text(bm_value);

    createLineGraph(
      "#sharpeChart",
      processed_data,
      "Sharpe",
      min_value,
      max_value,
      "Date",
      "Sharpe Ratio",
      0
    );
  }
);
