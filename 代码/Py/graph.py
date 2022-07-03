
class Graph():
  """
  对应论文4.6.1
  主要功能是:
    1.把结点添加入图
    2.画图
  """
  def __init__(self):
      self.nodes=[]

  def clear_jacobi(self):
      for node in self.nodes:
          node.clear_jacobi()

  def add_node(self,node):
      self.nodes.append(node)

  def nodes_count(self):
      # 返回图里的node的个数
      return len(self.nodes)

  def draw(self, ax=None):
      """
      画图
      """
      try:
          import networkx as nx
          import matplotlib.pyplot as plt
          from matplotlib.colors import ListedColormap
          import numpy as np
      except:
          raise Exception("Need Module networkx")

      G = nx.Graph()
      already = []
      labels = {}
      for node in self.nodes:
          G.add_node(node)
          # 如果该结点是变量结点，则标签为 类名+形状；如果为运算结点，则为类名+范数

          if node.name is not  None:
              labels[node]=node.name
          else:
              labels[node] = node.__class__.__name__ + (
                  "({:s})".format(str(node.size)) if hasattr(node, "size") else "") \
                             + ("\n[{:.3f}]".format(np.linalg.norm(node.jacobi))
                                if node.jacobi is not None else "")
          for c in node.children :
              if {node, c} not in already:
                  G.add_edge(node, c)
                  already.append({node, c})

      if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)

      ax.clear()
      ax.axis("on")  #显示坐标轴、坐标轴标签
      ax.grid(True)

      pos = nx.spring_layout(G, seed=42)   #布局的美化，减少布局的交叉边

      # edgecolors是结点的边框颜色

      # 有雅克比的变量节点, 权值结点，边框颜色是紫罗兰色#EE82EE
      cm = plt.cm.Reds
      nodelist = [n for n in self.nodes if n.__class__.__name__ ==
                  "Variable" and n.jacobi is not None]
      colorlist = [np.linalg.norm(n.jacobi) for n in nodelist]
      nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color="#FAEBD7", cmap=cm, edgecolors="#EE82EE",
                             node_size=2000, alpha=1.0, ax=ax)

      # 无雅克比的变量节点，边框颜色是 巧克力色 #D2691E
      nodelist = [n for n in self.nodes if n.__class__.__name__ ==
                  "Variable" and n.jacobi is None]
      nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color="#FFFFE0", cmap=cm, edgecolors="#D2691E",
                             node_size=2000, alpha=1.0, ax=ax)

      # 有雅克比的计算节点，边框颜色是黄色#FFFF00
      nodelist = [n for n in self.nodes if n.__class__.__name__ !=
                  "Variable" and n.jacobi is not None]
      colorlist = [np.linalg.norm(n.jacobi) for n in nodelist]
      nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color= "#F0F8FF", cmap=cm, edgecolors="#FFFF00",
                             node_size=2000, alpha=1.0, ax=ax)

      # 无雅克比的中间 ，即不参与反向传播的节点，边框颜色是深绿色#006400
      nodelist = [n for n in self.nodes if n.__class__.__name__ !=
                  "Variable" and n.jacobi is None]
      nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color="#FFB6C1", cmap=cm, edgecolors="#006400",
                             node_size=2000, alpha=1.0, ax=ax)

      # 边
      nx.draw_networkx_edges(G, pos, width=2, edge_color="#FFA500", ax=ax)
      nx.draw_networkx_labels(G, pos, labels=labels, font_weight="bold", font_color="#6c6c6c", font_size=8,
                              font_family='arial', ax=ax)

      # 保存图像
      plt.savefig("computing_graph.png")  # save as png

  def get_node_from_graph(node_name, name_scope=None, graph=None):

      if graph is None: graph = default_graph
      if name_scope: node_name = name_scope + '/' + node_name
      for node in graph.nodes:
          if node.name == node_name:
              return node
      return None



# 全局默认计算图
default_graph = Graph()
