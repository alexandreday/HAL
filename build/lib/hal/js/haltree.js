// Tree plotting
plotDiv = document.getElementById("plt1")
var plot_width = plotDiv.clientWidth;
var plot_height = plotDiv.clientHeight;
var idx_merge;

// Let's clean this up !
d3.json("tree.json", function(error1, treeData) { // read tree data => some information is not good  
    d3.json("tsne.json",function(error2, tSNEdata){ // read t-SNE data -> maybe we can do better here
        d3.json("idx_merge.json", function(error3, idx_merge_){
        if (error1) throw error1;
        if (error2) throw error2;
        if (error3) throw error3;

    var plot_info = {1:"median_markers",2:"feature_importance"};
    var title_list = {1:"Median marker expression",2:"Feature importance score"};
    //var plot_info = ["median_markers","feature_importance"];
    var plot_type = ["barchart", "scatter"];
    var tsne_is_rendered = false;
    var node_label = "cv";
    idx_merge = idx_merge_;

    var plot_choice = ['plt1','plt2'];
    //var title_list = ["Median marker expression","Feature importance score"];

    // Read in feature names
    var nestedTree = treeData['nestedTree']
    if(treeData['feature_name'].length == 0){
        var feature_name = range(0, treeData["median_markers"].length);
    }
    else{
        var feature_name = treeData['feature_name'];
    };
    
    var node_radius = 20;
    var node_color_default = "white", node_color_select = "#6acef2";
    var depth_height = 100;
    var dilate_spacing = 1.0;

    /* var chartDiv = document.getElementById("dragscroll"); // Adjustable !
    var width = chartDiv.clientWidth;
    var height = chartDiv.clientHeight; */

    var width = 1000;// in the future, should set those according to max width and depth of tree
    var height = 1000; 

    // Set the dimensions and margins of the diagram
    var margin = {top: 50, right: 50, bottom: 30, left: 80};
    /* width = width - margin.left - margin.right,
    height = height - margin.top - margin.bottom; */


var svg = d3.select("body #tree")
    .append("svg").attr("width",3.5*width).attr("height",2.0*height) // the rest of drawing should not exceed this bounding box
    //.attr("width", width + margin.right + margin.left)
    //.attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate("+ margin.left + "," + margin.top + ")");

svg.append("text").attr("class","plot-title").attr("transform", "translate(" + 250 + " ," + -20 + ")")
          .style("text-anchor", "middle")
          .text("Hierarchical Structure")

var i = 0, duration = 100;

/* Initialize tooltip */
tip = d3.tip().attr('class', 'd3-tip').html(function(d) { 
    var out = `<b>name</b> ${'&nbsp;'.repeat(1)}${d.data.name}<br>
				<b>info</b> ${'&nbsp;'.repeat(1)}${d.data.info}<br>`
    //var out = d.data.name + "<br>" + d.data.info;
    return out; }).offset([0, 12]).direction('e');

// declares a tree layout and assigns the size
var treemap = d3.tree().size([width, height]);

// Assigns parent, children, height, depth
var root = d3.hierarchy(nestedTree, function(d) { return d.children; });
/* root.x0 = width / 2;
root.y0 = 0; */
root.x0 = 0; // does not change anything somehow
root.y0 = height/2;
//console.log(root);

/* Invoke the tip in the context of your visualization */
svg.call(tip);

/* // Collapse after the second level
root.children.forEach(collapse); */
var menu1 = d3.selectAll("#menu1.custom-select");
var menu2 = d3.selectAll("#menu2.custom-select");


var render_plot=false, click_node_info=root, click_node;
var old_selection = {'menu1':[0,false],'menu2':[0,false]};
var selection = {'menu1':[0,false],'menu2':[0,false]}; // variable keeping track of selections and rendering
var options = {"Plot option:\n":0, "Median expression\n":1, "Feature importance\n":2, "t-SNE\n":3}

menu1.on("click", function(d){
    var t = d3.select(this).selectAll("select-selected");
    var option_selected = t._parents[0]["innerText"];
    update_menu('menu1', 0, option_selected)
});

menu2.on("click", function(d){
    var t = d3.select(this).selectAll("select-selected");
    var option_selected = t._parents[0]["innerText"];
    update_menu('menu2', 1, option_selected)
});

function update_menu(name, pos, option_selected){

    var current_selection = selection[name][0];
    
    if(option_selected!=current_selection){
        var opt = options[option_selected];

        old_selection[name] = selection[name]; // save previous selection
        selection[name] = [opt, true]; // update selection

        if(opt == 1 || opt == 2){
            barchart(feature_name, click_node_info.data[plot_info[opt]], title_list[opt], plot_choice[pos])
        }
        else if(opt == 3){
            scatter(tSNEdata["x"], tSNEdata["y"], tSNEdata["idx"], title_list[opt], plot_choice[pos])
        }
    }
}

update(root);

function update(source) {

  // Assigns the x and y position for the nodes
  var treeData = treemap(root);


  // Compute the new tree layout.
  var nodes = treeData.descendants(),
      links = treeData.descendants().slice(1);

  // Normalize for fixed-depth.
  nodes.forEach(function(d){ d.y = d.depth * depth_height; d.x = dilate_spacing*d.x;});

  // ****************** Nodes section ***************************

  // Update the nodes...
  var node = svg.selectAll('g.node')
      .data(nodes, function(d) {return d.id || (d.id = ++i); });

  // Enter any new modes at the parent's previous position.
  var nodeEnter = node.enter().append('g')
      .attr('class', 'node')
      .attr("transform", function(d) {
        return "translate(" + source.y0 + "," + source.x0 + ")";
    })

  // Add Circle for the nodes
  nodeEnter.append('circle')
      .attr('class', 'node')
      .attr('r', 1e-6)
      .style("fill", function(d) {
          return d._children ? "lightsteelblue" : "#fff";
      });

  // Add labels for the nodes
  nodeEnter.append('text')
      .attr('class','node-text')
      .attr("dy", ".35em")
      .attr("x", 0)
      .attr("text-anchor", "middle")
      .text(function(d) {return d.data[node_label];}) // - - === Save this === - - //
      .attr('cursor', 'pointer');

  // UPDATE 
  var nodeUpdate = nodeEnter.merge(node);

    nodeUpdate.attr("transform", function(d) { 
        return "translate(" + d.y + "," + d.x + ")";
        });

  // Update the node attributes and style
  nodeUpdate.select('circle.node')
    .attr('r', node_radius)
    .style("fill", function(d) {
        return d._children ? "lightsteelblue" : "#fff";
    })

    nodeUpdate.select('circle.node').
    on('click', function(d){
        // Check which tick box is clicked
        //console.log("here1");
        if(typeof click_node != 'undefined'){
            click_node.transition().duration(250).attr('r',node_radius).style("fill",node_color_default);
            if(d3.select(this).property("__data__")==click_node.property("__data__")){ // double clicking node
                d3.select(this).transition().duration(100).attr('r', node_radius).style('fill', node_color_default);
                return;
            }
        }
        
        click_node = d3.select(this); // save current node
        click_node_info = click_node._groups[0][0].__data__

        var menu_ls = ['menu1','menu2']
        for(var i=0;i<2;i++){
            menu = menu_ls[i]

            old_selection[menu] = selection[menu]
            opt = selection[menu][0]

            if(opt == 1 || opt == 2){
                barchart(feature_name, click_node_info.data[plot_info[opt]], title_list[opt], plot_choice[i])
            }
            else if(opt == 3){
                var node_name = +click_node_info.data['name']
                if(old_selection[menu][0] != 3){
                    scatter(tSNEdata["x"], tSNEdata["y"], tSNEdata["idx"], title_list[opt], plot_choice[i]);
                    restyle(tSNEdata["idx"], node_name, plot_choice[i]);
                }
                else{
                    restyle(tSNEdata["idx"], node_name, plot_choice[i]);
                }
            }
        }
        click_node.transition().duration(250).attr('r',node_radius*3.0).style("fill",node_color_select);
    })

    nodeUpdate.select('circle.node')
    .on('mouseover',
        function(d){
            // This is triggered only if node is different than current selection
            cd = d3.select(this)
            //console.log("clicknode")
            if(typeof click_node == 'undefined'){
                cd.transition().duration(100).style("fill", "brown").attr('r', node_radius*2.0);
            }
            else if(click_node._groups[0][0].__data__ != cd._groups[0][0].__data__){
                cd.transition().duration(100).style("fill", "brown").attr('r', node_radius*2.0);
            };
            tip.show(d);
        })
    .on('mouseout',function(d){
        cd = d3.select(this)
        // Trigger this only if node is not clicked
        if(typeof click_node == 'undefined'){
            d3.select(this).transition().duration(100).attr('r', node_radius).style('fill', node_color_default);
        }
        else if(click_node._groups[0][0].__data__ != cd._groups[0][0].__data__){
            d3.select(this).transition().duration(100).attr('r', node_radius).style('fill', node_color_default)
        }
        tip.hide(d);
    })


  // Remove any exiting nodes
  var nodeExit = node.exit()
        //.transition()
      //.duration(duration)
      .attr("transform", function(d) {
          return "translate(" + source.y + "," + source.x + ")";
      })
      .remove();

  // On exit reduce the node circles size to 0
  nodeExit.select('circle')
    .attr('r', 1e-6);

  // On exit reduce the opacity of text labels
  nodeExit.select('text')
    .style('fill-opacity', 1e-6);

  // ****************** links section ***************************

  // Update the links...
  var link = svg.selectAll('path.link')
      .data(links, function(d) { return d.id; });

  // Enter any new links at the parent's previous position.
  var linkEnter = link.enter().insert('path', "g")
      .attr("class", "link")
      .attr('d', function(d){
        var o = {x: source.y0, y: source.x0}
        return diagonal(o, o)
      });

  // UPDATE
  var linkUpdate = linkEnter.merge(link);

  // Transition back to the parent element position
  linkUpdate//.transition().ease(d3.easeSin)
      //.duration(duration)
      .attr('d', function(d){ return diagonal(d, d.parent) });


  // Remove any exiting links
  /* var linkExit = link.exit().transition()
      .duration(duration)
      .attr('d', function(d) {
        var o = {x: source.x, y: source.y}
        return diagonal(o, o)
      })
      .remove(); */

  // Store the old positions for transition.
  nodes.forEach(function(d){
    d.x0 = d.y;
    d.y0 = d.x;
  });

  // Creates a curved (diagonal) path from parent to the child nodes
  function diagonal(s, d) { 
    // (hint) These are like the bars you get for curves in keynote
    path = `M ${s.y} ${s.x}
    C ${(s.y + d.y) / 2} ${s.x},
      ${(s.y + d.y) / 2} ${d.x},
      ${d.y} ${d.x}`

    /* path = `M ${s.y} ${s.x}
            C ${s.y} ${(s.x + d.x)/2}, 
              ${d.x} ${(s.y + d.y)/2},
              ${d.y} ${d.x}` */

    return path
  }

  // Toggle children on click.
  function click(d) {
    if (d.children) {
        d._children = d.children;
        d.children = null;
      } else {
        d.children = d._children;
        d._children = null;
      }
    update(d);
  }
}
})})});

function range(start, end) {
    var foo = [];
    for (var i = start; i <= end; i++) {
        foo.push(i);
    }
    return foo;
}

function barchart(x_, y_, title, pos="plt1"){
    var trace1 = {
        x: x_, // feature names
        y: y_,
        type: 'bar',
        marker: {
          color: 'rgb(16, 88, 204)',
          line: {
            color:'rgb(0,0,0)',
            width:0.5
        }
        }
    };
    var data = [trace1];

    var layout = {
        font: {
            family: 'Helvetica',
            size: 14,
            color: '#7f7f7f'
        },
        showlegend: false,
        xaxis: {
          tickangle: -45
        },
        yaxis: {
          zeroline: true,
          gridwidth: 2
        },
        autosize: false,
        width: plot_width,
        height: plot_height,
        margin: {
            l: 0.08*plot_width,
            r: 0.08*plot_width,
            b: 0.25*plot_width,
            t: 0.1*plot_width
        },
        bargap :0.05,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor :'rgba(0,0,0,0)'
      };
    Plotly.newPlot(pos, data, layout, {displaylogo: false, showLink:false});
}

var color_list = ['rgb(48, 162, 218)', 'rgb(252, 79, 48)', 'rgb(229, 174, 56)', 'rgb(109, 144, 79)', 'rgb(139, 139, 139)', 'rgb(0, 111, 166)', 'rgb(163, 0, 89)', 'rgb(175, 141, 195)', 'rgb(146, 35, 41)', 'rgb(30, 110, 0)', 'rgb(255, 52, 255)', 'rgb(255, 74, 70)', 'rgb(0, 137, 65)', 'rgb(0, 111, 166)', 'rgb(163, 0, 89)', 'rgb(0, 0, 166)', 'rgb(99, 255, 172)', 'rgb(183, 151, 98)', 'rgb(0, 77, 67)', 'rgb(143, 176, 255)', 'rgb(153, 125, 135)', 'rgb(90, 0, 7)', 'rgb(128, 150, 147)', 'rgb(27, 68, 0)', 'rgb(79, 198, 1)', 'rgb(59, 93, 255)', 'rgb(74, 59, 83)', 'rgb(255, 47, 128)', 'rgb(97, 97, 90)', 'rgb(186, 9, 0)', 'rgb(107, 121, 0)', 'rgb(0, 194, 160)', 'rgb(255, 170, 146)', 'rgb(255, 144, 201)', 'rgb(185, 3, 170)', 'rgb(209, 97, 0)', 'rgb(221, 239, 255)', 'rgb(0, 0, 53)', 'rgb(123, 79, 75)', 'rgb(161, 194, 153)', 'rgb(48, 0, 24)', 'rgb(10, 166, 216)', 'rgb(1, 51, 73)', 'rgb(0, 132, 111)', 'rgb(55, 33, 1)', 'rgb(255, 181, 0)', 'rgb(194, 255, 237)', 'rgb(160, 121, 191)', 'rgb(204, 7, 68)', 'rgb(192, 185, 178)', 'rgb(194, 255, 153)', 'rgb(0, 30, 9)', 'rgb(0, 72, 156)', 'rgb(111, 0, 98)', 'rgb(12, 189, 102)', 'rgb(238, 195, 255)', 'rgb(69, 109, 117)', 'rgb(183, 123, 104)', 'rgb(122, 135, 161)', 'rgb(120, 141, 102)', 'rgb(136, 85, 120)', 'rgb(250, 208, 159)', 'rgb(255, 138, 154)', 'rgb(209, 87, 160)', 'rgb(190, 196, 89)', 'rgb(69, 102, 72)', 'rgb(0, 134, 237)', 'rgb(136, 111, 76)', 'rgb(52, 54, 45)', 'rgb(180, 168, 189)', 'rgb(0, 166, 170)', 'rgb(69, 44, 44)', 'rgb(99, 99, 117)', 'rgb(163, 200, 201)', 'rgb(255, 145, 63)', 'rgb(147, 138, 129)', 'rgb(87, 83, 41)', 'rgb(0, 254, 207)', 'rgb(176, 91, 111)', 'rgb(140, 208, 255)', 'rgb(59, 151, 0)', 'rgb(4, 247, 87)', 'rgb(200, 161, 161)', 'rgb(30, 110, 0)', 'rgb(121, 0, 215)', 'rgb(167, 117, 0)', 'rgb(99, 103, 169)', 'rgb(160, 88, 55)', 'rgb(107, 0, 44)', 'rgb(119, 38, 0)', 'rgb(215, 144, 255)', 'rgb(155, 151, 0)', 'rgb(84, 158, 121)', 'rgb(255, 246, 159)', 'rgb(32, 22, 37)', 'rgb(114, 65, 143)', 'rgb(188, 35, 255)', 'rgb(153, 173, 192)', 'rgb(58, 36, 101)', 'rgb(146, 35, 41)', 'rgb(91, 69, 52)', 'rgb(253, 232, 220)', 'rgb(64, 78, 85)', 'rgb(0, 137, 163)', 'rgb(203, 126, 152)', 'rgb(164, 232, 4)', 'rgb(50, 78, 114)', 'rgb(106, 58, 76)', 'rgb(122, 73, 0)'];

function translate(z){
    var col = new Array(z.length);
    for(var i=0;i<z.length;i++){
        if(z[i] < 0){
            col[i] = 'rgba(114, 255, 247, 0.3)';
        }
        else{
            col[i]=color_list[z[i]];
        };
    }
    return col;
}

function scatter(x_, y_, z_, title, pos="plt1"){

    // Goal, on click get proper markers and color the corresponding points in red

    //color_z = translate(z_);

    var trace1 = {
        x: x_,
        y: y_,
        text:[],
        mode: 'markers',
        hoverinfo:'none',
        type:'scattergl',
        marker: {
          color: translate(z_),
          size:10,
        } 
    };

    var data = [trace1];

    var layout = {
        hovermode:'off',
        font: {
            family: 'Helvetica',
            size: 14,
            color: '#7f7f7f'
        },
        showlegend: false,
        xaxis: {
          tickangle: -45
        },
        yaxis: {
          zeroline: true,
          gridwidth: 2
        },

        width: plot_width,
        height: plot_height,
        margin: {
            l: 0.08*plot_width,
            r: 0.08*plot_width,
            b: 0.12*plot_width,
            t: 0.05*plot_width
        },
        bargap :0.05,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor :'rgba(0,0,0,0)'
      };
    Plotly.newPlot(pos, data, layout, {displaylogo: false, showLink:false});
    };

function makeArrayOf(value, length) {
    var arr = [], i = length;
    while (i--) {
      arr[i] = value;
    }
    return arr;
  }

function restyle(idx, new_idx,pos="plt1"){
    idx_within = idx_merge[new_idx]
    var highlight_color = 'rgb(119, 255, 167)'
    //console.log(idx_within)
    var color = new Array(idx.length);
    var opacity = new Array(idx.length);
    var Qcolor_red;
    for(var i=0;i<idx.length;i++){
        Qcolor_red=false;
        for(var j=0;j<idx_within.length;j++){
            if(idx[i] == idx_within[j]){
                Qcolor_red=true;
            }
        }
        if(Qcolor_red){
            opacity[i] = 1.0;
            //color[i] = highlight_color;
        }
        else{
            opacity[i] = 0.05;
        };
    }
    //Plotly.restyle(pos, 'marker.color',[color]) // everything stays the same except transparency !
    Plotly.restyle(pos, 'marker.opacity',[opacity]);
}