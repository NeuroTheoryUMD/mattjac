// First undefine 'circles' so we can easily reload this file.
require.undef('grids');

define('grids', ['d3'], function (d3) {
    function draw(container, layers, edges, width, height, layer_dims, padding, layer_padding) {
    
        // TODO: remove this bad global var
        var cells = [];
    
        function gridData(layers) {
            console.log(layers);
            console.log(edges);
            
            var id = 0;
            var data = new Array();
            for (var layer = 0; layer < layers.length; layer++) {
                // get the rows and cols for this layer
                var rows = layer_dims[layer].rows;
                var cols = layer_dims[layer].cols;
            
                console.log('==LAYER==',layer);
                var boxes = layers[layer]; // get the boxes out of the layer
                var num_boxes = boxes.length;
                
                var layer_height = height/layers.length - layer_padding;
                console.log('layer_height=',layer_height);
                var width_minus_padding = (width-(padding*(cols-1)));
                console.log('layer_padding='+layer_padding);
                var layer_height_minus_padding = layer_height - (padding*(rows-1));
                console.log('width_minus_padding = ' + width_minus_padding);
                console.log('heigt_minus_padding = ' + layer_height_minus_padding);
                
                var box = 0;
                // iterate over rows
                for (var row = 0; row < rows; row++) {
                    for (var col = 0; col < cols; col++) {
                        var sqrows = boxes[0].length;
                        var sqcols = boxes[0][0].length;
                        console.log('row='+row+', col='+col);
                        var ypos = row*(layer_height/rows) + layer*layer_height + layer*layer_padding;
                        var xpos = col*(width/cols);
                        
                        console.log('x',xpos,'y',ypos);
                        
                        for (var sqrow = 0; sqrow < sqrows; sqrow++) {
                            for (var sqcol = 0; sqcol < sqcols; sqcol++) {
                                var w = width_minus_padding/cols/sqcols;
                                var h = layer_height_minus_padding/rows/sqrows;
                                console.log('w='+w+',h='+h);
                                var click = 0;
                                var val = boxes[box][sqrow][sqcol];
                                
                                var props = {
                                    x: xpos,
                                    y: ypos,
                                    width: w,
                                    height: h,
                                    click: click,
                                    id: "c"+id,
                                    assoc: edges["c"+id],
                                    color: "rgb("+255*val+","+255*val+","+255*val+")",
                                    val: val
                                };
                                
                                data.push(new Array());
                                data[row].push(props)
                                cells.push(props); // push all the cells into a flat array for convenience 
                                
                                // increment the x position. I.e. move it over by w
                                xpos += w;                                
                                id++; // incremenent the global id for the next cell
                            }
                            // reset the x position after a row is complete
                            xpos = col * width/cols;
                            // increment the y position for the next row. Move it down by h
                            ypos += h;
                        }
                        box++; // increment the box we are looking at
                    }
                }
            }
            return data;
        }
    
        var gridData = gridData(layers);	
    
        // set default width and height
        //width = width || 400;
        //height = height || 200;
        
        // create the grid svg 
        var grid = d3.select(container).append("svg")
            .attr('width', width)
            .attr('height', height);
        
        var row = grid.selectAll(".row")
        	.data(gridData)
        	.enter().append("g")
        	.attr("class", "row");
        
        var column = row.selectAll(".square")
        	.data(function(d) { return d; })
        	.enter().append("rect")
        	.attr("id", function(d) { return d.id; })
        	.attr("class","square")
        	.attr("x", function(d) { return d.x; })
        	.attr("y", function(d) { return d.y; })
        	.attr("width", function(d) { return d.width; })
        	.attr("height", function(d) { return d.height; })
        	.style("fill", function(d) { return d.color; })
        	//.style("stroke", "#222")
        	.on('mouseover', function(d) {
        	    console.log('d', d);
                for (var c=0; c < d.assoc.length; c++) {
                    d3.select('#c'+d.assoc[c])
                      .transition('fade').duration(200)
                      .attr("opacity", 1.0)
                      .style("fill", "rgb(255,0,0)");
                }
            })
            .on('mouseout', function(d) {
                for (var c=0; c < d.assoc.length; c++) {
                    d3.select('#c'+d.assoc[c])
                      .transition('fade').duration(200)
                      .attr("opacity", 1.0)
                      .style("fill", cells[c].color);
                }
            });
    }

    return draw;
});

element.append('<small>&#x25C9; &#x25CB; &#x25EF; Loaded gridex.js &#x25CC; &#x25CE; &#x25CF;</small>');