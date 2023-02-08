// First undefine 'circles' so we can easily reload this file.
require.undef('plot');

define('plot', ['d3', 'chartjs'], function (d3, chart) {
    function draw(container, data) {
        console.log('something');
        console.log(data);
        
        var canvases = d3.select("#myChart")
          .selectAll('canvas') // select the nonexistent ps and make them
          .data(data)
          .enter()
          .append('div')
          .attr("id", function(d) { return d; });
        console.log(canvases);

        document.getElementById('#1').innerHTML = '<canvas id="chart"></canvas>';
        const ctx = document.getElementById("#chart");
        var mychart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
          datasets: [{
            label: '# of Votes',
            data: [12, 19, 3, 5, 2, 3],
            borderWidth: 1
          }]
        },
        options: {
          scales: {
            y: {
              beginAtZero: true
            }
          }
        }
        });
        mychart.update();
    }

    return draw;
});

console.log('something2');
element.append('<small>&#x25C9; &#x25CB; &#x25EF; Loaded plot.js &#x25CC; &#x25CE; &#x25CF;</small>');