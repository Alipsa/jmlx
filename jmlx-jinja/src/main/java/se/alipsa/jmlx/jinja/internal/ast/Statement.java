package se.alipsa.jmlx.jinja.internal.ast;

import java.util.List;
import java.util.Objects;
import se.alipsa.jmlx.jinja.SourceLocation;

/** Statement nodes ported from upstream {@code ast.ts}. */
public sealed interface Statement
    permits Statement.Program,
        Statement.If,
        Statement.For,
        Statement.Break,
        Statement.Continue,
        Statement.SetStatement,
        Statement.Macro,
        Statement.Comment,
        Statement.FilterStatement,
        Statement.CallStatement,
        Expression {
  /**
   * Returns where this node starts in the source.
   *
   * @return the start location
   */
  SourceLocation location();

  /**
   * A whole parsed template.
   *
   * @param body the top-level statements, in source order
   * @param location the start of the template
   */
  record Program(List<Statement> body, SourceLocation location) implements Statement {
    /** Rejects a null {@code location} and defensively copies {@code body}. */
    public Program {
      body = List.copyOf(body);
      Objects.requireNonNull(location);
    }
  }

  /**
   * A conditional statement.
   *
   * @param test the branch condition
   * @param body statements to run when {@code test} is truthy
   * @param alternate statements to run otherwise; may contain a single nested {@code If} for an
   *     {@code elif}, or the {@code else} body, or be empty
   * @param location the start of the {@code if} tag
   */
  record If(
      Expression test, List<Statement> body, List<Statement> alternate, SourceLocation location)
      implements Statement {
    /** Rejects a null {@code test} or {@code location} and defensively copies both lists. */
    public If {
      Objects.requireNonNull(test);
      body = List.copyOf(body);
      alternate = List.copyOf(alternate);
      Objects.requireNonNull(location);
    }
  }

  /**
   * An iteration statement.
   *
   * @param loopVariable the identifier or tuple bound on each iteration
   * @param iterable the expression producing the values to iterate over
   * @param body statements run once per iterated value
   * @param defaultBlock statements run instead when {@code iterable} is empty ({@code for ...
   *     else})
   * @param location the start of the {@code for} tag
   */
  record For(
      Expression loopVariable,
      Expression iterable,
      List<Statement> body,
      List<Statement> defaultBlock,
      SourceLocation location)
      implements Statement {
    /**
     * Rejects a null {@code loopVariable}, {@code iterable}, or {@code location} and defensively
     * copies both lists.
     */
    public For {
      Objects.requireNonNull(loopVariable);
      Objects.requireNonNull(iterable);
      body = List.copyOf(body);
      defaultBlock = List.copyOf(defaultBlock);
      Objects.requireNonNull(location);
    }
  }

  /**
   * A loop break statement.
   *
   * @param location the start of the {@code break} tag
   */
  record Break(SourceLocation location) implements Statement {
    /** Rejects a null {@code location}. */
    public Break {
      Objects.requireNonNull(location);
    }
  }

  /**
   * A loop continue statement.
   *
   * @param location the start of the {@code continue} tag
   */
  record Continue(SourceLocation location) implements Statement {
    /** Rejects a null {@code location}. */
    public Continue {
      Objects.requireNonNull(location);
    }
  }

  /**
   * An assignment or block-capture statement; {@code value} is null for block capture.
   *
   * @param assignee the identifier or tuple being assigned
   * @param value the assigned expression, or null when this is a block-capture ({@code set ...
   *     endset})
   * @param body the captured block's statements; empty unless this is a block-capture
   * @param location the start of the {@code set} tag
   */
  record SetStatement(
      Expression assignee, Expression value, List<Statement> body, SourceLocation location)
      implements Statement {
    /** Rejects a null {@code assignee} or {@code location} and defensively copies {@code body}. */
    public SetStatement {
      Objects.requireNonNull(assignee);
      body = List.copyOf(body);
      Objects.requireNonNull(location);
    }
  }

  /**
   * A macro declaration.
   *
   * @param name the macro's identifier
   * @param args the declared parameters, including any default-value expressions
   * @param body the macro's statements
   * @param location the start of the {@code macro} tag
   */
  record Macro(
      Expression.Identifier name,
      List<Expression> args,
      List<Statement> body,
      SourceLocation location)
      implements Statement {
    /** Rejects a null {@code name} or {@code location} and defensively copies both lists. */
    public Macro {
      Objects.requireNonNull(name);
      args = List.copyOf(args);
      body = List.copyOf(body);
      Objects.requireNonNull(location);
    }
  }

  /**
   * A template comment.
   *
   * @param value the comment's text, excluding the surrounding comment delimiters
   * @param location the start of the comment
   */
  record Comment(String value, SourceLocation location) implements Statement {
    /** Rejects a null {@code value} or {@code location}. */
    public Comment {
      Objects.requireNonNull(value);
      Objects.requireNonNull(location);
    }
  }

  /**
   * A block passed through a filter.
   *
   * @param filter the filter identifier or call applied to the rendered body
   * @param body the statements to render before filtering
   * @param location the start of the {@code filter} tag
   */
  record FilterStatement(Expression filter, List<Statement> body, SourceLocation location)
      implements Statement {
    /** Rejects a null {@code filter} or {@code location} and defensively copies {@code body}. */
    public FilterStatement {
      Objects.requireNonNull(filter);
      body = List.copyOf(body);
      Objects.requireNonNull(location);
    }
  }

  /**
   * A caller block; {@code callerArgs} is null when the call declares no caller arguments.
   *
   * @param call the macro call this block invokes
   * @param callerArgs the parameters exposed to the macro's {@code caller()}, or null if none
   * @param body the statements available to the macro via {@code caller()}
   * @param location the start of the {@code call} tag
   */
  record CallStatement(
      Expression.CallExpression call,
      List<Expression> callerArgs,
      List<Statement> body,
      SourceLocation location)
      implements Statement {
    /**
     * Rejects a null {@code call} or {@code location} and defensively copies {@code callerArgs}
     * (when non-null) and {@code body}.
     */
    public CallStatement {
      Objects.requireNonNull(call);
      callerArgs = callerArgs == null ? null : List.copyOf(callerArgs);
      body = List.copyOf(body);
      Objects.requireNonNull(location);
    }
  }
}
