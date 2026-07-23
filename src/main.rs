use rayon::prelude::*;
use std::sync::mpsc;
use std::fs::File;
use std::io::{Write, BufWriter};
use std::env;
use std::thread;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <width(n)> <height(m)> <num_samples_to_collect>", args[0]);
        std::process::exit(1);
    }

    let width: usize = args[1].parse().unwrap();
    let height: usize = args[2].parse().unwrap();
    let num_samples: usize = args[3].parse().unwrap();
    let num_nodes = width * height;

    if num_nodes % 2 != 0 {
        eprintln!("Error: Grid must have an even number of vertices.");
        std::process::exit(1);
    }
    
    // 1. Generate the Grid Adjacency List directly in Rust (Row-Major)
    let mut adj = vec![Vec::with_capacity(4); num_nodes];
    for y in 0..height {
        for x in 0..width {
            let id = y * width + x;
            if x > 0 { adj[id].push(id - 1); } // left
            if x < width - 1 { adj[id].push(id + 1); } // right
            if y > 0 { adj[id].push(id - width); } // up
            if y < height - 1 { adj[id].push(id + width); } // down
        }
    }

    println!("Generated {}x{} grid ({} nodes). Searching for {} samples...", width, height, num_nodes, num_samples);

    // 2. Setup a channel for worker threads to send epsilons to the main writer thread
    let (tx, rx) = mpsc::sync_channel::<(f64, usize)>(100);
    
    // 3. Spawn a dedicated writer thread to write epsilons to a file as they arrive
    let writer_thread = thread::spawn(move || {
        let filename = format!("epsilons{}x{}-{}.txt", width, height, num_samples);
        let file = File::create(&filename).expect("Could not create output file");

        let mut writer = BufWriter::new(file);
        let mut count = 0;
        
        while let Ok((eps, boundary_length)) = rx.recv() {
            count += 1;
            writeln!(writer, "{}, {}", eps, boundary_length).unwrap();
            if count % 10 == 0 {
                println!("Collected {} / {} samples...", count, num_samples);
            }
            if count >= num_samples {
                break;
            }
        }
        println!("Finished collecting {} samples. Saved to {}", num_samples, filename);
    });

    // 4. Parallel Worker Pool
    let target_size = num_nodes / 2;
    let num_threads = rayon::current_num_threads();
    
    rayon::scope(|s| {
        for thread_id in 0..num_threads {
            let tx = tx.clone();
            let adj = &adj;

            s.spawn(move |_| {
                let mut rng = fastrand::Rng::new();
                rng.seed(thread_id as u64 ^ std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64);

                let mut in_tree = vec![0u32; num_nodes];
                let mut next_node = vec![0usize; num_nodes];
                let mut tree_adj = vec![Vec::with_capacity(4); num_nodes];
                
                let mut parents = vec![0usize; num_nodes];
                let mut order = vec![0usize; num_nodes];
                let mut subtree_sizes = vec![1usize; num_nodes];
                let mut in_v1 = vec![false; num_nodes];
                let mut q = vec![0usize; num_nodes];
                
                let mut generation = 1u32;

                loop {
                    // Stop this thread if the channel is closed (meaning we hit our sample target)
                    // if tx.send(0.0).is_err() { break; } 
                    
                    for edges in tree_adj.iter_mut() { edges.clear(); }

                    // A. Wilson's Algorithm
                    in_tree[0] = generation; 
                    for i in 1..num_nodes {
                        if in_tree[i] == generation { continue; }
                        let mut u = i;
                        while in_tree[u] != generation {
                            let neighbors = &adj[u];
                            next_node[u] = neighbors[rng.usize(0..neighbors.len())];
                            u = next_node[u];
                        }
                        u = i;
                        while in_tree[u] != generation {
                            in_tree[u] = generation;
                            let nxt = next_node[u];
                            tree_adj[u].push(nxt);
                            tree_adj[nxt].push(u);
                            u = nxt;
                        }
                    }

                    // B. Check for Bisection
                    if let Some((u, p)) = check_bisection(&tree_adj, &mut parents, &mut order, &mut subtree_sizes, target_size) {
                        
                        // C. Calculate Epsilon exactly as defined in the math problem
                        in_v1.fill(false);
                        in_v1[u] = true;
                        
                        let mut head = 0;
                        let mut tail = 1;
                        q[0] = u;
                        
                        // BFS to find all nodes strictly in component V1
                        while head < tail {
                            let curr = q[head];
                            head += 1;
                            for &nxt in &tree_adj[curr] {
                                if nxt != p && !in_v1[nxt] {
                                    in_v1[nxt] = true;
                                    q[tail] = nxt;
                                    tail += 1;
                                }
                            }
                        }
                        
                        // Find bounds of the interface boundary B1 U B2
                        let mut min_x = width;
                        let mut max_x = 0;
                        let mut boundary_length = 0;
                        
                        for i in 0..num_nodes {
                            if in_v1[i] {
                                for &neighbor in &adj[i] {
                                    if !in_v1[neighbor] {
                                        boundary_length += 1;
                                        
                                        // Both i and neighbor are on the boundary!
                                        let x1 = i % width;
                                        let x2 = neighbor % width;
                                        min_x = min_x.min(x1).min(x2);
                                        max_x = max_x.max(x1).max(x2);
                                    }
                                }
                            }
                        }
                        
                        let center = (width as f64 - 1.0) / 2.0;
                        let deviation = (center - min_x as f64).max(max_x as f64 - center);
                        let eps = deviation / (height as f64);
                        
                        // Send successful epsilon to the writer thread (ignore if closed)
                        // Send successful epsilon to the writer thread and stop if the channel is closed
                        if tx.send((eps, boundary_length)).is_err() { 
                            break; 
                        }
                    }

                    generation = generation.wrapping_add(1);
                    if generation == 0 {
                        in_tree.fill(0);
                        generation = 1;
                    }
                }
            });
        }
    });
    
    writer_thread.join().unwrap();
}

fn check_bisection(
    tree_adj: &[Vec<usize>], 
    parents: &mut [usize], 
    order: &mut [usize], 
    subtree_sizes: &mut [usize], 
    target: usize
) -> Option<(usize, usize)> {
    let n = tree_adj.len();
    let mut head = 0;
    let mut tail = 0;
    
    parents[0] = usize::MAX; 
    order[tail] = 0;
    tail += 1;
    
    while head < tail {
        let curr = order[head];
        head += 1;
        for &nxt in &tree_adj[curr] {
            if nxt != parents[curr] {
                parents[nxt] = curr;
                order[tail] = nxt;
                tail += 1;
            }
        }
    }
    
    subtree_sizes.fill(1);
    for i in (1..n).rev() {
        let u = order[i];
        let p = parents[u];
        if subtree_sizes[u] == target {
            return Some((u, p));
        }
        subtree_sizes[p] += subtree_sizes[u];
    }
    None
}