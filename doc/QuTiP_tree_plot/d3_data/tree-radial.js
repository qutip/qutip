var r = 1500;

var tree = d3.layout.tree()
    .size([360, r - 175])
    .separation(function(a, b) { return (a.parent == b.parent ? 1 : 2) / a.depth; });

var diagonal = d3.svg.diagonal.radial()
    .projection(function(d) { return [d.y, d.x / 180 * Math.PI]; });

var vis = d3.select("#chart").append("svg")
    .attr("width", r * 2 )
    .attr("height", r * 2 )
    .append("g")
    .attr("transform", "translate(" + r + "," + r + ")");

d3.json("d3_data/qutip.json", function(json) {
  var nodes = tree.nodes(json);

  var link = vis.selectAll("path.link")
      .data(tree.links(nodes))
      .enter().append("path")
      .attr("class", "link")
      .attr("d", diagonal);

  var node = vis.selectAll("g.node")
      .data(nodes)
      .enter().append("g")
      .attr("class", "node")
      .attr("transform", function(d) { return "rotate(" + (d.x - 90) + ")translate(" + d.y + ")"; })

  node.append("circle")
      .attr("r", 2.5)
      /*.style("fill", function(d) { return d.color; })*/
	  .style("stroke", function(d) { return d.color; });

  node.append("text")
      .attr("dx", function(d) { return d.x < 180 ? 8 : -8; })
      .attr("dy", ".31em")
      .attr("text-anchor", function(d) { return d.x < 180 ? "start" : "end"; })
      .attr("transform", function(d) { return d.x < 180 ? null : "rotate(180)"; })
	  .style("fill", function(d) { return d.color; })
      .text(function(d) { return d.name; });
});
