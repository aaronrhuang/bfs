// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <iostream>
#include <vector>

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "sliding_queue.h"
#include "timer.h"
#include "compressed_graph.h"


/*
GAP Benchmark Suite
Kernel: Breadth-First Search (BFS)
Author: Scott Beamer

Will return parent array for a BFS traversal from a source vertex

This BFS implementation makes use of the Direction-Optimizing approach [1].
It uses the alpha and beta parameters to determine whether to switch search
directions. For representing the frontier, it uses a SlidingQueue for the
top-down approach and a Bitmap for the bottom-up approach. To reduce
false-sharing for the top-down approach, thread-local QueueBuffer's are used.

To save time computing the number of edges exiting the frontier, this
implementation precomputes the degrees in bulk at the beginning by storing
them in parent array as negative numbers. Thus the encoding of parent is:
  parent[x] < 0 implies x is unvisited and parent[x] = -out_degree(x)
  parent[x] >= 0 implies x been visited

[1] Scott Beamer, Krste Asanović, and David Patterson. "Direction-Optimizing
    Breadth-First Search." International Conference on High Performance
    Computing, Networking, Storage and Analysis (SC), Salt Lake City, Utah,
    November 2012.
*/


using namespace std;

int64_t BUStep(const CompGraph &g, pvector<NodeID> &parent, Bitmap &front, Bitmap &next) {

  int64_t awake_count = 0;
  next.reset();
#pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
  for (NodeID u=0; u < g.num_nodes(); u++) {
    if (parent[u] < 0 && g.in_degree(u) > 0) {

      // Handle first vertex
      NodeID *neigh = g.in_neigh_start(u);
      NodeID v = *(neigh++);

      // Attempt computation for first vertex
      if (front.get_bit(v)) {
        parent[u] = v;
        awake_count++;
        next.set_bit(u);
        continue;
      }

      // Handle deltas
      auto offsetReader = (vertexOffset *) neigh;
      for (int i = 1; i < g.in_degree(u); ++i) {
        NodeID value_read = *(offsetReader++);
        if (value_read < (NodeID)MAX_OFFSET) {
          v += value_read;
        } else {
          auto nodeReader = (NodeID *) offsetReader;
          v = *(nodeReader++);
          offsetReader = (vertexOffset *) nodeReader;
        }


        if (front.get_bit(v)) {
          parent[u] = v;
          awake_count++;
          next.set_bit(u);
          break;
        }
      }
    }
  }
  return awake_count;
}


int64_t TDStep(const CompGraph &g, pvector<NodeID> &parent, SlidingQueue<NodeID> &queue) {
  int64_t scout_count = 0;
#pragma omp parallel
  {
    QueueBuffer<NodeID> lqueue(queue);
#pragma omp for reduction(+ : scout_count)
    for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
      NodeID u = *q_iter;

      // Handle first vertex
      if (g.out_degree(u) > 0) {

        NodeID *neigh = g.out_neigh_start(u);
        NodeID v = *(neigh++);

        // Make updates for first vertex
        NodeID val = parent[v];
        if (val < 0) {
          if (compare_and_swap(parent[v], val, u)) {
            lqueue.push_back(v);
            scout_count += -val;
          }
        }

        // Handle deltas
        auto offsetReader = (vertexOffset *) neigh;
        for (int i = 1; i < g.out_degree(u); ++i) {
          NodeID value_read = *(offsetReader++);
          if (value_read < (NodeID)MAX_OFFSET) {
            v += value_read;
          } else {
            auto nodeReader = (NodeID *) offsetReader;
            v = *(nodeReader++);
            offsetReader = (vertexOffset *) nodeReader;
          }

          // Make updates
          NodeID curr_val = parent[v];
          if (curr_val < 0) {
            if (compare_and_swap(parent[v], curr_val, u)) {
              lqueue.push_back(v);
              scout_count += -curr_val;
            }
          }
        }
      }
    }

    lqueue.flush();
  }
  return scout_count;
}


void QueueToBitmap(const SlidingQueue<NodeID> &queue, Bitmap &bm) {
#pragma omp parallel for
  for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
    NodeID u = *q_iter;
    bm.set_bit_atomic(u);
  }
}

void BitmapToQueue(const CompGraph &g, const Bitmap &bm,
                   SlidingQueue<NodeID> &queue) {
#pragma omp parallel
  {
    QueueBuffer<NodeID> lqueue(queue);
#pragma omp for
    for (NodeID n=0; n < g.num_nodes(); n++)
      if (bm.get_bit(n))
        lqueue.push_back(n);
    lqueue.flush();
  }
  queue.slide_window();
}

pvector<NodeID> InitParent(const CompGraph &g) {
  pvector<NodeID> parent(g.num_nodes());
#pragma omp parallel for
  for (NodeID n=0; n < g.num_nodes(); n++)
    parent[n] = g.out_degree(n) != 0 ? -g.out_degree(n) : -1;
  return parent;
}

pvector<NodeID> DOBFS(const CompGraph &g, NodeID source, int alpha = 15, int beta = 18) {
  PrintStep("Source", static_cast<int64_t>(source));
  Timer t;
  t.Start();
  pvector<NodeID> parent = InitParent(g);
  t.Stop();
  PrintStep("i", t.Seconds());
  parent[source] = source;
  SlidingQueue<NodeID> queue(g.num_nodes());
  queue.push_back(source);
  queue.slide_window();
  Bitmap curr(g.num_nodes());
  curr.reset();
  Bitmap front(g.num_nodes());
  front.reset();
  int64_t edges_to_check = g.num_edges_directed();
  int64_t scout_count = g.out_degree(source);
  while (!queue.empty()) {
    if (scout_count > edges_to_check / alpha) {
      int64_t awake_count, old_awake_count;
      TIME_OP(t, QueueToBitmap(queue, front));
      PrintStep("e", t.Seconds());
      awake_count = queue.size();
      queue.slide_window();
      do {
        t.Start();
        old_awake_count = awake_count;
        awake_count = BUStep(g, parent, front, curr);
        front.swap(curr);
        t.Stop();
        PrintStep("bu", t.Seconds(), awake_count);
      } while ((awake_count >= old_awake_count) ||
               (awake_count > g.num_nodes() / beta));
      TIME_OP(t, BitmapToQueue(g, front, queue));
      PrintStep("c", t.Seconds());
      scout_count = 1;
    } else {
      t.Start();
      edges_to_check -= scout_count;
      scout_count = TDStep(g, parent, queue);
      queue.slide_window();
      t.Stop();
      PrintStep("td", t.Seconds(), queue.size());
    }
  }
#pragma omp parallel for
  for (NodeID n = 0; n < g.num_nodes(); n++)
    if (parent[n] < -1)
      parent[n] = -1;
  return parent;
}


void PrintBFSStats(const CompGraph &g, const pvector<NodeID> &bfs_tree) {
  int64_t tree_size = 0;
  int64_t n_edges = 0;
  for (NodeID n : g.vertices()) {
    if (bfs_tree[n] >= 0) {
      n_edges += g.out_degree(n);
      tree_size++;
    }
  }
  cout << "BFS Tree has " << tree_size << " nodes and ";
  cout << n_edges << " edges" << endl;
}


// BFS verifier does a serial BFS from same source and asserts:
// - parent[source] = source
// - parent[v] = u  =>  depth[v] = depth[u] + 1 (except for source)
// - parent[v] = u  => there is edge from u to v
// - all vertices reachable from source have a parent
bool BFSVerifier(const Graph &g, NodeID source,
                 const pvector<NodeID> &parent) {
  pvector<int> depth(g.num_nodes(), -1);
  depth[source] = 0;
  vector<NodeID> to_visit;
  to_visit.reserve(g.num_nodes());
  to_visit.push_back(source);
  for (auto it = to_visit.begin(); it != to_visit.end(); it++) {
    NodeID u = *it;
    for (NodeID v : g.out_neigh(u)) {
      if (depth[v] == -1) {
        depth[v] = depth[u] + 1;
        to_visit.push_back(v);
      }
    }
  }
  for (NodeID u : g.vertices()) {
    if ((depth[u] != -1) && (parent[u] != -1)) {
      if (u == source) {
        if (!((parent[u] == u) && (depth[u] == 0))) {
          cout << "Source wrong" << endl;
          return false;
        }
        continue;
      }
      bool parent_found = false;
      for (NodeID v : g.in_neigh(u)) {
        if (v == parent[u]) {
          if (depth[v] != depth[u] - 1) {
            cout << "Wrong depths for " << u << " & " << v << endl;
            return false;
          }
          parent_found = true;
          break;
        }
      }
      if (!parent_found) {
        cout << "Couldn't find edge from " << parent[u] << " to " << u << endl;
        return false;
      }
    } else if (depth[u] != parent[u]) {
      cout << "Reachability mismatch" << endl;
      cout << "Expected " << depth[u] << " but got " << parent[u] << " for node " << u << endl;
      return false;
    }
  }
  return true;
}

std::vector<NodeID> uncompress_row(const CompGraph &dg, int64_t row, bool out_deg) {

  std::vector<NodeID> vec_ids;
  NodeID* neigh = out_deg? dg.out_neigh_start(row) : dg.in_neigh_start(row);
  NodeID current_node = *neigh;
  vec_ids.push_back(current_node);
  ++neigh;


  vertexOffset* offsetReader = (vertexOffset *)neigh;
  for(int64_t i = 0; i < dg.out_degree(row); ++i) {

    NodeID value_read = *offsetReader;
    if (value_read == MAX_OFFSET) {
      ++offsetReader;
      NodeID * nodeReader = (NodeID *)offsetReader;
      current_node = *nodeReader;
      vec_ids.push_back(current_node);
      ++nodeReader;
      offsetReader = (vertexOffset *)nodeReader;
    } else {
      current_node += value_read;
      vec_ids.push_back(current_node);
      ++offsetReader;
    }
  }

  return vec_ids;
}

void compareGraphs(const Graph &g, const CompGraph &dg) {

  for(int64_t i = 0; i < g.num_nodes(); ++i) {

    if (g.out_degree(i) != dg.out_degree(i) ){
      std::cout << "Out degree for row " << i << " incorrect. Got " << dg.out_degree(i) <<
                   " but expected " << g.out_degree(i) << std::endl;
      exit(1);
    }

    if (g.in_degree(i) != dg.in_degree(i) ){
      std::cout << "In degree for row " << i << " incorrect. Got " << dg.in_degree(i) <<
                " but expected " << g.in_degree(i) << std::endl;
      exit(1);
    }

    // Check out neighbors
    std::vector<NodeID> row_list = uncompress_row(dg, i, true);
    std::vector<NodeID> correct_ids;
    for(NodeID v : g.out_neigh(i)) {
      correct_ids.push_back(v);
    }

    for(int j = 0; j < g.out_degree(i); ++j) {
      if (row_list[j] != correct_ids[j] ) {
        std::cout << "Mismatch in compression - decompression at row " << i << " got " << row_list[j] << " but "
                  << correct_ids[j] << " was expected at location " << j << std::endl;
        exit(1);
      }
    }

    // Check in neighbors
    std::vector<NodeID> in_row_list = uncompress_row(dg, i, false);
    std::vector<NodeID> in_correct_ids;
    for(NodeID v : g.in_neigh(i)) {
      in_correct_ids.push_back(v);
    }

    for(int j = 0; j < g.in_degree(i); ++j) {
      if (in_row_list[j] != in_correct_ids[j] ) {
        std::cout << "Mismatch in compression - decompression at col " << i << " got " << in_row_list[j] << " but "
                  << in_correct_ids[j] << " was expected at location " << j << std::endl;
        exit(1);
      }
    }
  }



}

int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "breadth-first search");
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  CompGraph delta_g = b.MakeDeltaGraph();

  SourcePicker<CompGraph> sp(delta_g, cli.start_vertex());
  auto BFSBound = [&sp] (const CompGraph &delta_g) { return DOBFS(delta_g, sp.PickNext()); };

  SourcePicker<CompGraph> vsp(delta_g, cli.start_vertex());
  auto VerifierBound = [&vsp, &b] (const CompGraph &delta_g, const pvector<NodeID> &parent) {
    Graph csr_g = b.MakeGraph();
    compareGraphs(csr_g, delta_g);
    return BFSVerifier(csr_g, vsp.PickNext(), parent);
  };

  BenchmarkKernel(cli, delta_g, BFSBound, PrintBFSStats, VerifierBound);
  return 0;
}
