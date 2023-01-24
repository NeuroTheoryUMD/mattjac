// First undefine 'circles' so we can easily reload this file.
require.undef('images');

define('images', ['d3'], function (d3) {

    function draw(container, data, width, height) {
        width = width || 600;
        height = height || 200;
        var svg = d3.select(container).append("svg")
            .attr('width', width)
            .attr('height', height)
            .append("g");

        var image = svg.selectAll('image').data(data);

        image.enter()
            .append('image')
            .attr("xlink:href", function (d) {return d;})
            .on('mouseover', function() {
                d3.select(this)
                  .transition('fade').duration(500)
                  .attr("opacity", 1);
            })
            .on('mouseout', function() {
                d3.select(this)
                    .transition('fade').duration(500)
                    .attr("opacity", 0.1);
            })
            .transition('fade').duration(2000)
            .attr("opacity", 1);
    }

    return draw;
});

element.append('<small>&#x25C9; &#x25CB; &#x25EF; Loaded images.js &#x25CC; &#x25CE; &#x25CF;</small>');
