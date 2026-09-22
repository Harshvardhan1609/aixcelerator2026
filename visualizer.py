import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional

def create_3d_vector_space_plot(
    engine,
    query: Optional[str] = None,
    retrieved_items: Optional[List[Dict[str, Any]]] = None
) -> go.Figure:
    """
    Projects ChromaDB high-dimensional embeddings (384-D) into an interactive 3D vector space
    using PCA, plotting all knowledge base chunks and highlighting the query & nearest neighbors.
    """
    data = engine.collection.get(include=["embeddings", "metadatas", "documents"])
    embeddings = np.array(data["embeddings"])
    metadatas = data["metadatas"]
    documents = data["documents"]
    ids = data["ids"]

    if len(embeddings) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Not enough chunks indexed for 3D projection.", showarrow=False)
        return fig

    # Fit PCA with 3 components
    pca = PCA(n_components=3, random_state=42)
    coords_3d = pca.fit_transform(embeddings)

    # Prepare labels and hover text
    top_ids = set()
    if retrieved_items:
        top_ids = {item["id"] for item in retrieved_items}

    # Group chunks into categories
    categories = [m.get("category", "General") for m in metadatas]
    unique_categories = list(dict.fromkeys(categories))
    
    # Palette tailored to SIN Brand (Imperial Crimson, Amber, Coral, Rose, Ruby)
    brand_colors = ["#EF4444", "#F59E0B", "#F87171", "#EC4899", "#FB923C", "#E11D48", "#A855F7"]
    cat_color_map = {cat: brand_colors[i % len(brand_colors)] for i, cat in enumerate(unique_categories)}

    fig = go.Figure()

    # 1. Plot background chunks grouped by category
    for cat in unique_categories:
        indices = [i for i, c in enumerate(categories) if c == cat and ids[i] not in top_ids]
        if not indices:
            continue
        
        x = coords_3d[indices, 0]
        y = coords_3d[indices, 1]
        z = coords_3d[indices, 2]
        hover_texts = [
            f"<b>{metadatas[i].get('title', ids[i])}</b><br>"
            f"<i>Category:</i> {cat}<br>"
            f"<i>Chunk ID:</i> {ids[i]}<br>"
            f"<br>{documents[i][:180]}..."
            for i in indices
        ]

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+text',
            name=cat[:22],
            text=[metadatas[i].get('title', ids[i])[:14] for i in indices],
            textposition="top center",
            textfont=dict(size=9, color="#9CA3AF"),
            marker=dict(
                size=6,
                color=cat_color_map[cat],
                opacity=0.75,
                line=dict(color="#1A0606", width=1)
            ),
            hovertext=hover_texts,
            hoverinfo="text"
        ))

    # 2. Highlight Top-K Retrieved Chunks
    if retrieved_items:
        top_indices = [i for i, chunk_id in enumerate(ids) if chunk_id in top_ids]
        if top_indices:
            x_top = coords_3d[top_indices, 0]
            y_top = coords_3d[top_indices, 1]
            z_top = coords_3d[top_indices, 2]
            top_hover = [
                f"<b>🎯 TOP RETRIEVED: {metadatas[i].get('title', ids[i])}</b><br>"
                f"<i>Category:</i> {categories[i]}<br>"
                f"<i>Rank:</i> #{[r['id'] for r in retrieved_items].index(ids[i]) + 1}<br>"
                f"<br>{documents[i][:220]}..."
                for i in top_indices
            ]

            fig.add_trace(go.Scatter3d(
                x=x_top, y=y_top, z=z_top,
                mode='markers+text',
                name='🎯 Top Matches',
                text=[f"#{[r['id'] for r in retrieved_items].index(ids[i]) + 1} {metadatas[i].get('title', ids[i])[:14]}" for i in top_indices],
                textposition="bottom center",
                textfont=dict(size=11, color="#FCA5A5"),
                marker=dict(
                    size=12,
                    color="#DC2626",
                    symbol="diamond",
                    line=dict(color="#FFFFFF", width=2),
                    opacity=0.95
                ),
                hovertext=top_hover,
                hoverinfo="text"
            ))

    # 3. If query provided, embed and project query point into 3D
    if query:
        query_emb = np.array(engine.embedding_fn([query]))
        query_3d = pca.transform(query_emb)[0]
        qx, qy, qz = query_3d[0], query_3d[1], query_3d[2]

        # Draw connecting vector rays from Query to Top-K items
        if retrieved_items:
            for rank, item in enumerate(retrieved_items, 1):
                chunk_id = item["id"]
                if chunk_id in ids:
                    idx = ids.index(chunk_id)
                    cx, cy, cz = coords_3d[idx, 0], coords_3d[idx, 1], coords_3d[idx, 2]
                    
                    fig.add_trace(go.Scatter3d(
                        x=[qx, cx], y=[qy, cy], z=[qz, cz],
                        mode='lines',
                        line=dict(color='rgba(245, 158, 11, 0.7)', width=3, dash='dash'),
                        showlegend=(rank == 1),
                        name='⚡ Retrieval Vector Rays',
                        hoverinfo='text',
                        hovertext=f"Nearest Neighbor Ray #{rank} -> {metadatas[idx].get('title', chunk_id)}"
                    ))

        # Query point (glowing star)
        fig.add_trace(go.Scatter3d(
            x=[qx], y=[qy], z=[qz],
            mode='markers+text',
            name='⭐ Student Query',
            text=["⭐ Query Vector"],
            textposition="top center",
            textfont=dict(size=13, color="#FDE68A"),
            marker=dict(
                size=14,
                color="#F59E0B",
                symbol="circle",
                line=dict(color="#FFFFFF", width=3),
                opacity=1.0
            ),
            hovertext=f"<b>Student Query:</b><br>\"{query}\"<br><br><i>Embedded into 384-D & projected to 3D</i>",
            hoverinfo="text"
        ))

    # Calculate variance explained
    explained_var = pca.explained_variance_ratio_ * 100
    total_var = sum(explained_var)

    fig.update_layout(
        title=dict(
            text=f"<b>ChromaDB 3D Vector Space</b> (PCA Variance Preserved: {total_var:.1f}%)",
            font=dict(family="Outfit, sans-serif", size=17, color="#FFFFFF"),
            x=0.02, y=0.96
        ),
        paper_bgcolor="#0D0606",
        plot_bgcolor="#0D0606",
        scene=dict(
            xaxis=dict(
                title=dict(text=f"PCA 1 ({explained_var[0]:.1f}%)", font=dict(color="#9CA3AF", size=11)),
                backgroundcolor="#160808",
                gridcolor="rgba(143, 0, 0, 0.25)",
                showbackground=True,
                zerolinecolor="rgba(255, 255, 255, 0.2)",
                tickfont=dict(color="#6B7280", size=9)
            ),
            yaxis=dict(
                title=dict(text=f"PCA 2 ({explained_var[1]:.1f}%)", font=dict(color="#9CA3AF", size=11)),
                backgroundcolor="#160808",
                gridcolor="rgba(143, 0, 0, 0.25)",
                showbackground=True,
                zerolinecolor="rgba(255, 255, 255, 0.2)",
                tickfont=dict(color="#6B7280", size=9)
            ),
            zaxis=dict(
                title=dict(text=f"PCA 3 ({explained_var[2]:.1f}%)", font=dict(color="#9CA3AF", size=11)),
                backgroundcolor="#160808",
                gridcolor="rgba(143, 0, 0, 0.25)",
                showbackground=True,
                zerolinecolor="rgba(255, 255, 255, 0.2)",
                tickfont=dict(color="#6B7280", size=9)
            ),
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.2)
            )
        ),
        legend=dict(
            font=dict(color="#D1D5DB", size=10),
            bgcolor="rgba(24, 7, 7, 0.8)",
            bordercolor="rgba(143, 0, 0, 0.4)",
            borderwidth=1,
            x=0.01, y=0.01
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig

def create_similarity_ranking_chart(retrieved_items: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders a clean horizontal bar chart displaying similarity confidence for retrieved chunks.
    """
    if not retrieved_items:
        fig = go.Figure()
        fig.add_annotation(text="No items retrieved yet.", showarrow=False)
        return fig

    titles = []
    scores = []
    hover_notes = []

    for item in retrieved_items:
        title = item.get("metadata", {}).get("title", item["id"])
        # Format distance into similarity percentage
        dist = item.get("distance", 0.5)
        sim_pct = max(0.0, min(100.0, (1.0 - (dist / 2.0)) * 100))
        
        titles.append(title[:26])
        scores.append(round(sim_pct, 1))
        hover_notes.append(
            f"<b>{title}</b><br>"
            f"Relevance: {sim_pct:.1f}%<br>"
            f"Distance: {dist:.4f}<br>"
            f"<br>{item.get('document', '')[:160]}..."
        )

    # Invert so rank 1 is at top
    titles.reverse()
    scores.reverse()
    hover_notes.reverse()

    fig = go.Figure(go.Bar(
        x=scores,
        y=titles,
        orientation='h',
        marker=dict(
            color=scores,
            colorscale=[[0, '#8F0000'], [0.5, '#DC2626'], [1.0, '#F59E0B']],
            line=dict(color='#FFFFFF', width=1)
        ),
        hovertext=hover_notes,
        hoverinfo='text',
        text=[f"{s}%" for s in scores],
        textposition='inside',
        textfont=dict(color="#FFFFFF", size=11, family="Inter")
    ))

    fig.update_layout(
        title=dict(
            text="<b>Top-K Semantic Match Confidence (%)</b>",
            font=dict(family="Outfit, sans-serif", size=15, color="#FFFFFF")
        ),
        paper_bgcolor="#120606",
        plot_bgcolor="#120606",
        xaxis=dict(
            title=dict(text="Estimated Semantic Similarity %", font=dict(color="#9CA3AF", size=11)),
            range=[0, 100],
            gridcolor="rgba(143, 0, 0, 0.2)",
            tickfont=dict(color="#6B7280", size=10)
        ),
        yaxis=dict(
            tickfont=dict(color="#E5E7EB", size=11)
        ),
        margin=dict(l=10, r=20, t=40, b=30),
        height=260
    )

    return fig

def create_rag_pipeline_sankey_chart(query: str, retrieved_items: List[Dict[str, Any]], model_name: str) -> go.Figure:
    """
    Renders an interactive Sankey diagram showing end-to-end RAG data flow.
    """
    labels = [
        "1. Student Query",
        "2. ONNX Embedding (384-D)",
        "3. ChromaDB Vector Store",
        "4. Context Assembler"
    ]
    
    # Add retrieved sources
    source_indices = []
    for idx, item in enumerate(retrieved_items[:4]):
        t = item.get("metadata", {}).get("title", f"Chunk {idx+1}")[:18]
        labels.append(f"Doc: {t}")
        source_indices.append(len(labels) - 1)

    labels.append(f"5. LLM ({model_name[:16]})")
    llm_idx = len(labels) - 1

    labels.append("6. Counsellor Response")
    out_idx = len(labels) - 1

    sources = [0, 1, 2]
    targets = [1, 2, 3]
    values = [10, 10, 10]

    # Flow from Context Assembler to each retrieved chunk
    for s_idx in source_indices:
        sources.append(3)
        targets.append(s_idx)
        values.append(3)
        
        sources.append(s_idx)
        targets.append(llm_idx)
        values.append(3)

    sources.append(llm_idx)
    targets.append(out_idx)
    values.append(12)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=18,
            line=dict(color="#000000", width=0.5),
            label=labels,
            color=["#F59E0B", "#DC2626", "#8F0000", "#B91C1C"] + ["#E11D48"] * len(source_indices) + ["#7C3AED", "#10B981"]
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(185, 28, 28, 0.35)"
        )
    )])

    fig.update_layout(
        title=dict(
            text="<b>End-to-End RAG Architecture Data Flow</b>",
            font=dict(family="Outfit, sans-serif", size=15, color="#FFFFFF")
        ),
        paper_bgcolor="#120606",
        font=dict(color="#E5E7EB", size=11),
        margin=dict(l=10, r=10, t=40, b=20),
        height=280
    )

    return fig

def create_token_economics_chart(turns_data: List[Dict[str, Any]], token_limit: int = 25000) -> go.Figure:
    """
    Renders an interactive multi-metric chart showing tokens used vs tokens saved per turn,
    along with cumulative consumption compared to the session limit.
    """
    if not turns_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No conversation turns recorded yet.<br>Ask a question to view real-time token economics.",
            showarrow=False,
            font=dict(color="#9CA3AF", size=13)
        )
        fig.update_layout(
            paper_bgcolor="#120606",
            plot_bgcolor="#120606",
            height=300
        )
        return fig

    turn_labels = [f"T{t['turn']}: {t['query'][:18]}..." for t in turns_data]
    prompt_tokens = [t["prompt_tokens"] for t in turns_data]
    completion_tokens = [t["completion_tokens"] for t in turns_data]
    tokens_saved = [t["tokens_saved"] for t in turns_data]
    cum_tokens = [t["cumulative_tokens"] for t in turns_data]

    fig = go.Figure()

    # Bar: Prompt Tokens Consumed
    fig.add_trace(go.Bar(
        x=turn_labels,
        y=prompt_tokens,
        name="Prompt Tokens Used",
        marker_color="#DC2626",
        opacity=0.9
    ))

    # Bar: Completion Tokens Generated
    fig.add_trace(go.Bar(
        x=turn_labels,
        y=completion_tokens,
        name="Completion Tokens",
        marker_color="#F59E0B",
        opacity=0.9
    ))

    # Bar: Tokens Saved by Reverse Memory Architecture
    fig.add_trace(go.Bar(
        x=turn_labels,
        y=tokens_saved,
        name="⚡ RMA Tokens Saved",
        marker_color="#10B981",
        opacity=0.95
    ))

    # Line: Cumulative Tokens Consumed (Secondary Y)
    fig.add_trace(go.Scatter(
        x=turn_labels,
        y=cum_tokens,
        name="Cumulative Tokens",
        mode="lines+markers",
        line=dict(color="#60A5FA", width=3),
        marker=dict(size=7, color="#FFFFFF", line=dict(color="#60A5FA", width=2)),
        yaxis="y2"
    ))

    # Horizontal Line: Session Token Limit (Secondary Y)
    fig.add_hline(
        y=token_limit,
        line_dash="dot",
        line_color="#EF4444",
        line_width=2,
        annotation_text=f"Session Token Limit ({token_limit:,})",
        annotation_position="top right",
        annotation_font=dict(color="#F87171", size=10),
        yref="y2"
    )

    fig.update_layout(
        title=dict(
            text="<b>Token Economics & Credit Savings per Turn</b>",
            font=dict(family="Outfit, sans-serif", size=16, color="#FFFFFF")
        ),
        paper_bgcolor="#120606",
        plot_bgcolor="#120606",
        barmode="group",
        xaxis=dict(
            tickfont=dict(color="#D1D5DB", size=10),
            gridcolor="rgba(143, 0, 0, 0.2)"
        ),
        yaxis=dict(
            title=dict(text="Turn Tokens", font=dict(color="#D1D5DB", size=11)),
            gridcolor="rgba(143, 0, 0, 0.2)",
            tickfont=dict(color="#9CA3AF", size=10)
        ),
        yaxis2=dict(
            title=dict(text="Cumulative Tokens", font=dict(color="#60A5FA", size=11)),
            overlaying="y",
            side="right",
            tickfont=dict(color="#60A5FA", size=10),
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#E5E7EB", size=10),
            bgcolor="rgba(20, 6, 6, 0.8)"
        ),
        margin=dict(l=20, r=40, t=50, b=30),
        height=320
    )

    return fig

def create_reverse_memory_flow_chart(
    is_cache_hit: bool,
    similarity: float,
    tokens_saved: int,
    model_name: str
) -> go.Figure:
    """
    Renders an architectural comparison diagram between Standard Forward RAG
    and Reverse Conversational Memory Architecture (RMA).
    """
    if is_cache_hit:
        labels = [
            "1. User Query",
            "2. Semantic Memory Cache Check",
            f"3. High Match ({similarity*100:.1f}%)",
            "4. Instant Memory Response",
            "5. OpenRouter LLM (BYPASSED)"
        ]
        # Query -> Check -> High Match -> Instant Response (LLM Bypassed)
        sources = [0, 1, 2]
        targets = [1, 2, 3]
        values = [10, 10, 10]
        node_colors = ["#F59E0B", "#3B82F6", "#10B981", "#059669", "#4B5563"]
        flow_title = f"⚡ Reverse Memory Tier 1: Cache Hit (100% Tokens Saved: ~{tokens_saved:,} tokens)"
    else:
        labels = [
            "1. User Query",
            "2. Vector Memory Lookup",
            "3. Relevant Snippet Extracted",
            "4. Pruned RAG Context",
            f"5. LLM Inference ({model_name[:12]})",
            "6. Counsellor Response"
        ]
        sources = [0, 1, 2, 3, 4]
        targets = [1, 2, 3, 4, 5]
        values = [8, 8, 8, 8, 8]
        node_colors = ["#F59E0B", "#3B82F6", "#10B981", "#DC2626", "#8F0000", "#10B981"]
        flow_title = f"⚡ Reverse Memory Tier 2: Relevant Memory Retrieval ({tokens_saved:,} Tokens Saved)"

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=18,
            line=dict(color="#000000", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(16, 185, 129, 0.4)" if is_cache_hit else "rgba(220, 38, 38, 0.35)"
        )
    )])

    fig.update_layout(
        title=dict(
            text=f"<b>{flow_title}</b>",
            font=dict(family="Outfit, sans-serif", size=14, color="#FFFFFF")
        ),
        paper_bgcolor="#120606",
        font=dict(color="#E5E7EB", size=11),
        margin=dict(l=10, r=10, t=40, b=20),
        height=260
    )

    return fig
