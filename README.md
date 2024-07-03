# Money Laundering Detection using Graph Neural Networks

## Project Overview

This project aims to detect various patterns in money laundering using advanced Graph Neural Network (GNN) techniques

## Key Features

- Utilizes Graph Isomorphism Network (GIN) and Principal Neighborhood Aggregation (PNA) graph neural networks
- Processes large-scale synthetic transaction data (approximately 6,000,000 rows)
- Converts 8 key transaction features into embeddings
- Captures laundering patterns through graph convolutional layers

## Data Features

The model processes the following transaction features:
1. Sending account ID
2. Receiving account ID
3. Timestamp
4. Sending amount
5. Receiving amount
6. Sending currency
7. Receiving currency
8. Payment format

## Model Performance

The PNA model achieved the following performance metrics:
- F1 score: 0.51
- Precision: 0.65
- Recall: 0.40

## Technical Implementation

1. **Data Pre-processing**: Implemented a pipeline to prepare the large-scale transaction data for compatibility with the GNN model.

2. **GNN Architecture**: Utilized GIN and PNA architectures to convert transaction features into embeddings.

3. **Pattern Detection**: Employed graph convolutional layers to capture complex money laundering patterns within the transaction network.

## Future Work

- Explore additional GNN architectures for potential performance improvements.U
- Investigate methods to enhance the model's recall while maintaining high precision
- Analyze the model's performance on real-world transaction data

=
