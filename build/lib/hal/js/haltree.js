// Tree plotting
plotDiv = document.getElementById("plt1")
var plot_width = plotDiv.clientWidth;
var plot_height = plotDiv.clientHeight;
var idx_merge;

// Let's clean this up !
d3.json("tree.json", function(error1, treeData) { // read tree data => some information is not good  
    d3.json('tsne.json',function(error2, tSNEdata){ // read t-SNE data -> maybe we can do better here
        d3.json('idx_merge.json', function(error3, idx_merge_){
        if (error1) throw error1;
        if (error2) throw error2;
        if (error3) throw error3;

    var plot_info = ["median_markers","feature_importance"];
    var plot_type = ["barchart","scatter"];
    var tsne_is_rendered = false;
    var node_label = "cv";
    idx_merge = idx_merge_;
    var easement = "easeCubic";
    var plot_choice = ['plt1','plt2'];
    var title_list = ["Median marker expression","Feature importance score"];
    var marker_name = range(0, treeData["median_markers"].length);
    
    var node_radius = 20;
    var node_color_default = "white", node_color_select = "#6acef2";
    var depth_height = 100;

    /* var chartDiv = document.getElementById("dragscroll"); // Adjustable !
    var width = chartDiv.clientWidth;
    var height = chartDiv.clientHeight; */

    var width = 1000;// in the future, should set those according to max width and depth of tree
    var height = 1000; 

// Set the dimensions and margins of the diagram
    var margin = {top: 50, right: 50, bottom: 30, left: 50};
    /* width = width - margin.left - margin.right,
    height = height - margin.top - margin.bottom; */

// append the svg object to the body of the page
// appends a 'group' element to 'svg'
// moves the 'group' element to the top left margin
var svg = d3.select("body #tree")
    .append("svg").attr("width",1.5*width).attr("height",1.5*height) // the rest of drawing should not exceed this bounding box
    //.attr("width", width + margin.right + margin.left)
    //.attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate("+ margin.left + "," + margin.top + ")");

svg.append("text").attr("class","plot-title").attr("transform", "translate(" + 250 + " ," + -20 + ")")
          .style("text-anchor", "middle")
          .text("Hierarchical structure");

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
var root = d3.hierarchy(treeData, function(d) { return d.children; });
/* root.x0 = width / 2;
root.y0 = 0; */
root.x0 = 0;
root.y0 = height / 2;

/* Invoke the tip in the context of your visualization */
svg.call(tip);

/* // Collapse after the second level
root.children.forEach(collapse); */
var menu1 = d3.select("#menu1");
var menu2 = d3.select("#menu2");

menu1.selectAll(".myCheck").call(init);
menu2.selectAll(".myCheck").call(init);
//console.log(d3.select("body menu1 myCheck"))

var old_choice=[0,0], render_plot=false, click_node; // this is tricky .. !

menu1.selectAll(".myCheck").on("change", function(d){
    update_2(menu1,1);
});

menu2.selectAll(".myCheck").on("change", function(d){
    update_2(menu2, menu_number=2);
});
//console.log('Old choice\t'+old_choice)

function init(d){
    d._groups[0][0]["checked"]=true;
}

function update_2(menu, menu_number=1){
    var new_choice, i=0, old_choice_ = old_choice[menu_number-1];
    menu.selectAll(".myCheck").each(function(d){
      var cb = d3.select(this);
      if(cb.property("checked")){
        if(i!=old_choice_){new_choice = i;}
      }
      i++;
    });

    i=0;
    menu.selectAll(".myCheck").each(function(d){
      var cb = d3.select(this);
      if(cb.property("checked")){
        if(i==old_choice_){
            cb._groups[0][0]["checked"]=false;}
      }
      i++;
    });

    if(old_choice[menu_number-1] == 2){
        tsne_is_rendered=false;
    }
    old_choice[menu_number-1]=new_choice;

    if(render_plot){
        //@barchart(marker_name, d.data[plot_info[old_choice[1]]],title_list[old_choice[1]], "plt2")
        if(new_choice == 2){
            if(!tsne_is_rendered){
                scatter(tSNEdata["x"],tSNEdata["y"], tSNEdata["idx"], title_list[new_choice], plot_choice[menu_number-1])
                tsne_is_rendered =  true;
            }
            else{

                //console.log('restyling 2')
                restyle(tSNEdata["idx"], 1)
            }
        }
        else{
            barchart(marker_name, click_node._groups[0][0].__data__.data[plot_info[new_choice]], title_list[new_choice], plot_choice[menu_number-1])
        }
    }
}

update(root);


/* // Collapse the node and all it's children
function collapse(d) {
  if(d.children) {
    d._children = d.children
    d._children.forEach(collapse)
    d.children = null
  }
} */

function update(source) {

  // Assigns the x and y position for the nodes
  var treeData = treemap(root);


  // Compute the new tree layout.
  var nodes = treeData.descendants(),
      links = treeData.descendants().slice(1);

  // Normalize for fixed-depth.
  nodes.forEach(function(d){ d.y = d.depth * depth_height});

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

  // Transition to the proper position for the node
  //nodeUpdate
    //.transition().ease(d3.easeSin)
    //.duration(duration)
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
        }
        click_node = d3.select(this); // save current node
        // should be triggered if something is ticked
        render_plot = true;
        for(var i=0;i<2;i++){
            if(old_choice[i] == 2){
                //console.log('restyling')
                var node_name = +click_node._groups[0][0].__data__['data']['name']
                //click_node._groups[0][0].
                //console.log(click_node.selectAll('name'))
                restyle(tSNEdata["idx"], node_name, plot_choice[i])
                // here replot only new points // subset of data ... 
                //scatter(tSNEdata["x"],tSNEdata["y"],tSNEdata["idx"], title_list[old_choice[i]], plot_choice[i])
            }
            else{
                barchart(marker_name, d.data[plot_info[old_choice[i]]], title_list[old_choice[i]], plot_choice[i])
            }
        }

        /* barchart(marker_name, d.data[plot_info[old_choice[0]]],title_list[old_choice[0]])
        barchart(marker_name, d.data[plot_info[old_choice[1]]],title_list[old_choice[1]], "plt2") */
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
        x: x_,
        y: y_,
        type: 'bar',
        marker: {
          color: 'rgb(18, 209, 41)'
        }
    };
    var data = [trace1];

    var layout = {
        font:{
          family: 'Roboto Mono' 
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


var color_list = ["#30a2da","#fc4f30","#e5ae38","#6d904f","#8b8b8b","#006FA6", "#A30059","#af8dc3","#922329","#1E6E00","#FF34FF", "#FF4A46","#008941", "#006FA6", "#A30059", "#0000A6", "#63FFAC","#B79762", "#004D43", "#8FB0FF", "#997D87","#5A0007", "#809693", 
"#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80","#61615A", "#BA0900","#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100","#DDEFFF",
"#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F","#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99",
"#001E09","#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66","#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459",
"#456648", "#0086ED", "#886F4C","#34362D", "#B4A8BD", "#00A6AA", "#452C2C","#636375", "#A3C8C9", "#FF913F", "#938A81","#575329", "#00FECF", "#B05B6F",
"#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00","#7900D7", "#A77500","#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700","#549E79",
"#FFF69F", "#201625", "#72418F","#BC23FF","#99ADC0","#3A2465","#922329","#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C","#7A4900"];

function translate(z){
    var col = new Array(z.length);
    for(var i=0;i<z.length;i++){
        col[i]=color_list[z[i]];
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
        font:{
          family: 'Roboto Mono' 
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
            b: 0.1*plot_width,
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
    //console.log(idx_within)
    var color = new Array(idx.length);
    var Qcolor_red;
    for(var i=0;i<idx.length;i++){
        Qcolor_red=false;
        for(var j=0;j<idx_within.length;j++){
            if(idx[i] == idx_within[j]){
                Qcolor_red=true;
            }
        }
        if(Qcolor_red){
            color[i] = 'red';
        }
        else{
            color[i] = color_list[idx[i]];
        }
    }
    Plotly.restyle(pos, 'marker.color',[color])
}