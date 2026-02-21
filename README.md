A `Pending<R>` primitive paired with a `Responder<R>` primitive, which work in conjunction to allow you to return a value from a worker thread. Includes a `rayon` feature for using [rayon](https://docs.rs/rayon/latest/rayon/) to spawn the worker threads. The API is very straightforward and easy to use.

# Example
```rust
use std::{
    thread::sleep,
    time::Duration,
};

use pending::{
    Pending,
    Responder,
    strategy,
    spawn,
};

fn main() {
    let (pending, join_handle) = spawn::<strategy::Std, _, _>(|| {
        sleep(Duration::from_secs(3));
        0xDEADBEEFu32
    });
    
    sleep(Duration::from_millis(2750));
    
    if let Ok(result) = pending.try_recv() {
        println!("Result: 0x{result:0X}");
    } else {
        join_handle.join().expect("Failed to join.");
        let Ok(result) = pending.try_recv() else {
            panic!("Thread was joined with no response.");
        };
        println!("Result after join: 0x{result:0X}");
    }
}
```