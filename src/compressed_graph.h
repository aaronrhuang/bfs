#ifndef BFS_COMPRESSED_GRAPH_H
#define BFS_COMPRESSED_GRAPH_H

typedef uint16_t vertexOffset;
#define MAX_OFFSET ( (1u << 16u) - 1 )

template <class NodeID_, class DestID_ = NodeID_, bool MakeInverse = true>
class DeltaGraph {


  void ReleaseResources() {
    if (out_index_ != nullptr)
      delete[] out_index_;
    if (out_neighbors_ != nullptr)
      delete[] out_neighbors_;
    if (directed_) {
      if (in_index_ != nullptr)
        delete[] in_index_;
      if (in_neighbors_ != nullptr)
        delete[] in_neighbors_;
    }
  }


public:
  DeltaGraph() : directed_(false), num_nodes_(-1), num_edges_(-1),
               out_index_(nullptr), out_neighbors_(nullptr),
               in_index_(nullptr), in_neighbors_(nullptr) {}

  DeltaGraph(int64_t num_nodes, int64_t num_edges, DestID_* index, DestID_* neighs) :
          directed_(false), num_nodes_(num_nodes), num_edges_(num_edges),
          out_index_(index), out_neighbors_(neighs),
          in_index_(index), in_neighbors_(neighs) {}

  DeltaGraph(int64_t num_nodes, int64_t num_edges, DestID_* out_index, DestID_* out_neighs,
           DestID_* in_index, DestID_* in_neighs) :
          directed_(true), num_nodes_(num_nodes), num_edges_(num_edges),
          out_index_(out_index), out_neighbors_(out_neighs),
          in_index_(in_index), in_neighbors_(in_neighs) {}

  DeltaGraph(DeltaGraph&& other) : directed_(other.directed_),
                               num_nodes_(other.num_nodes_), num_edges_(other.num_edges_),
                               out_index_(other.out_index_), out_neighbors_(other.out_neighbors_),
                               in_index_(other.in_index_), in_neighbors_(other.in_neighbors_) {
    other.num_edges_ = -1;
    other.num_nodes_ = -1;
    other.out_index_ = nullptr;
    other.out_neighbors_ = nullptr;
    other.in_index_ = nullptr;
    other.in_neighbors_ = nullptr;
  }

  ~DeltaGraph() {
    ReleaseResources();
  }

  DeltaGraph& operator=(DeltaGraph&& other) {
    if (this != &other) {
      ReleaseResources();
      directed_ = other.directed_;
      num_edges_ = other.num_edges_;
      num_nodes_ = other.num_nodes_;
      out_index_ = other.out_index_;
      out_neighbors_ = other.out_neighbors_;
      in_index_ = other.in_index_;
      in_neighbors_ = other.in_neighbors_;
      other.num_edges_ = -1;
      other.num_nodes_ = -1;
      other.out_index_ = nullptr;
      other.out_neighbors_ = nullptr;
      other.in_index_ = nullptr;
      other.in_neighbors_ = nullptr;
    }
    return *this;
  }

  bool directed() const {
    return directed_;
  }

  int64_t num_nodes() const {
    return num_nodes_;
  }

  int64_t num_edges() const {
    return num_edges_;
  }

  int64_t num_edges_directed() const {
    return directed_ ? num_edges_ : 2*num_edges_;
  }

  int64_t out_degree(NodeID_ v) const {
    DestID_* base_ptr = (DestID_ *)((uint64_t)out_neighbors_ +  out_index_[v]);
    return *base_ptr;
  }

  int64_t in_degree(NodeID_ v) const {
    static_assert(MakeInverse, "Graph inversion disabled but reading inverse");
    DestID_* base_ptr = (DestID_ *)((uint64_t)in_neighbors_ +  in_index_[v]);
    return *base_ptr;
  }

  DestID_* out_neigh_start(NodeID_ n) const {
    return (DestID_ *)((uint64_t)out_neighbors_ +  out_index_[n] + sizeof(DestID_));
  }

  DestID_* in_neigh_start(NodeID_ n) const {
    static_assert(MakeInverse, "Graph inversion disabled but reading inverse");
    return (DestID_ *)((uint64_t)in_neighbors_ +  in_index_[n] + sizeof(DestID_));
  }

  void PrintStats() const {
    std::cout << "Graph has " << num_nodes_ << " nodes and "
              << num_edges_ << " ";
    if (!directed_)
      std::cout << "un";
    std::cout << "directed edges for degree: ";
    std::cout << num_edges_/num_nodes_ << std::endl;
  }

  void PrintTopology() const {
    for (NodeID_ i=0; i < num_nodes_; i++) {
      std::cout << i << ": ";
      for (DestID_ j : out_neigh(i)) {
        std::cout << j << " ";
      }
      std::cout << std::endl;
    }
  }

  Range<NodeID_> vertices() const {
    return Range<NodeID_>(num_nodes());
  }

private:
  bool directed_;
  int64_t num_nodes_;
  int64_t num_edges_;
  DestID_* out_index_;
  DestID_*  out_neighbors_;
  DestID_* in_index_;
  DestID_*  in_neighbors_;
};


#endif //BFS_COMPRESSED_GRAPH_H
