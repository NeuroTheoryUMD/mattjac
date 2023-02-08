// First undefine 'circles' so we can easily reload this file.
require.undef('grids');

define('grids', ['d3'], function (d3) {
    function draw(container, layers, edges, width, layer_heights, layer_dims, padding, layer_padding) {
    
        // TODO: remove this bad global var
        var cells = [];
        
        // calculate total layer height
        var height = 0;
        for (var layer in layers) {
            height += layer_heights[layer];
            height += layer_padding;
        }
        // remove padding at the end
        height -= layer_padding;
        
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
                 
                console.log(layer_heights);
                var layer_height = layer_heights[layer];
                console.log('layer_height=',layer_height);
                
                var sqrows = boxes[0].length;
                var sqcols = boxes[0][0].length;
                // figure out the max integer width to avoid using floats with pixels
                // computed_width = cols*col_width + (num_boxes-1)*padding
                var box_width = 1;
                console.log('sqcols',sqcols);
                console.log(cols*(box_width+1)*sqcols);
                while (cols*(box_width+1)*sqcols + (cols-1)*padding <= width) {
                    box_width += 1;
                    //console.log(cols*box_width*sqcols, (cols-1)*padding);
                }
                // TODO: don't need to compute these separately if aspect='square'
                // computed_layer_height = rows*row_height + (num_boxes-1)*padding
                var box_height = 1;
                while (rows*(box_height+1)*sqrows + (rows-1)*padding <= layer_height) {
                    box_height += 1;
                }
                
                console.log('box_width', box_width, 'box_height', box_height);
                
                var box = 0;
                // iterate over rows
                for (var row = 0; row < rows; row++) {
                    for (var col = 0; col < cols; col++) {
                        // stop adding new squares if we run out of them
                        if (box >= num_boxes) {
                            break;
                        }
                    
                        console.log('row='+row+', col='+col);
                        
                        // sum the previous layer_heights to determine the current layer starting position
                        var prev_layer_heights = 0;
                        for (var prev_layer = 0; prev_layer < layer; prev_layer++) {
                            prev_layer_heights += layer_heights[prev_layer];
                            prev_layer_heights += layer_padding;
                        }
                        
                        console.log('prev_layer_heights', prev_layer_heights);
                        var ypos = row*box_height*sqrows + row*padding + prev_layer_heights;
                        var xpos = col*box_width*sqcols + col*padding;
                        
                        console.log('x',xpos,'y',ypos);
                        
                        for (var sqrow = 0; sqrow < sqrows; sqrow++) {
                            for (var sqcol = 0; sqcol < sqcols; sqcol++) {
                                var w = box_width;
                                var h = box_height;
                                //console.log('w='+w+',h='+h);
                                var click = 0;
                                var val = boxes[box][sqrow][sqcol];
                                
                                var props = {
                                    x: xpos,
                                    y: ypos,
                                    width: w,
                                    height: h,
                                    click: click,
                                    id: "c"+id,
                                    assoc: edges[id],
                                    color: "rgb("+(127+128*val)+","+(127+128*val)+","+(127+128*val)+")",
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
                            xpos = col*box_width*sqcols + col*padding;
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
        	.on('mouseover', function(d) {
        	    // highlight the cell as well
        	    d3.select(this)
                    //.transition('fade').duration(200)
                    //.style("opacity", 1)
                    //.style('fill', 'rgb(255,0,0)')
                    //.attr("x", d.x - 2)
                    //.attr("y", d.y - 2)
                    //.attr("width", d.width + 4)
                    //.attr("height", d.height + 4)
                    .style("stroke", "rgb(255,0,0)");
        	    //console.log('d', d.assoc);
                for (var c in d.assoc) {
                    var weight = Math.round(127 + 128*d.assoc[c]*cells[c].val);
                    var reweighted_color = "rgb("+weight+","+weight+","+weight+")";
                    //console.log('c', c, 'w', weight, 'color', reweighted_color);
                    d3.select('#c'+c)
                        .transition('fade').duration(200)
                        .attr("opacity", 1.0)
                        //.style("fill", "rgb(255,0,0)");
                        .style("fill", reweighted_color);
                }
            })
            .on('mouseout', function(d) {
                // reset the cell's color 
                d3.select(this)
                    //.transition('fade').duration(200)
                    //.style("opacity", 1)
                    //.style('fill', d.color)
                    //.attr("x", d.x)
                    //.attr("y", d.y)
                    //.attr("width", d.width)
                    //.attr("height", d.height)
                    .style("stroke", "none");
                for (var c in d.assoc) {
                    d3.select('#c'+c)
                        .transition('fade').duration(200)
                        .attr("opacity", 1.0)
                        .style("fill", cells[c].color);
                }
            });
    }

    return draw;
});

element.append('<small>&#x25C9; &#x25CB; &#x25EF; Loaded grids.js &#x25CC; &#x25CE; &#x25CF;</small>');