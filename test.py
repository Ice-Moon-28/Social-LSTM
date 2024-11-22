import random

def generate_network_config(num_nodes=20, max_delay=0.050, prob=0.0):
    nodes = list(range(1, num_nodes + 1))
    config = []

    # Nodes
    config.append("[nodes]")
    config.append(" ".join(map(str, nodes)))

    # Links
    config.append("\n[links]")
    for i in range(1, num_nodes + 1):
        for j in range(i + 1, num_nodes + 1):
            delay = round(random.uniform(0.010, max_delay), 3)
            config.append(f"({i},{j}) delay {delay:.3f} prob {prob:.1f}")

    # Events
    config.append("\n[events]")
    event_times = [100, 110, 120, 150, 250, 300, 350, 440, 450, 460, 500, 510, 520, 530, 540, 550, 560, 600]
    event_types = ["xmit", "linkdying", "linkcomingup"]
    for t in event_times[:-1]:
        if random.random() < 0.6:  # 60% chance of xmit event
            src = random.randint(1, num_nodes)
            dst = random.randint(1, num_nodes)
            while dst == src:  # Ensure no self-loop
                dst = random.randint(1, num_nodes)
            config.append(f"{t:.2f} xmit ({src},{dst})")
        else:  # Link status change events
            link_type = random.choice(event_types[1:])
            node1 = random.randint(1, num_nodes)
            node2 = random.randint(1, num_nodes)
            while node2 == node1:  # Ensure no self-loop
                node2 = random.randint(1, num_nodes)
            config.append(f"{t:.2f} {link_type} ({min(node1, node2)},{max(node1, node2)})")
    
    config.append(f"{event_times[-1]:.2f} end")

    return "\n".join(config)

# Generate and save to file
config_text = generate_network_config(num_nodes=5, prob=0.1)
with open("network_config.txt", "w") as f:
    f.write(config_text)

print("Network configuration file 'network_config.txt' generated.")