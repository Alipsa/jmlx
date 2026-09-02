package se.alipsa.jmlx.jinja.internal.ast;

import java.util.List;
import java.util.Objects;
import se.alipsa.jmlx.jinja.SourceLocation;
import se.alipsa.jmlx.jinja.internal.lexer.Token;

/** Expression nodes ported from upstream {@code ast.ts}. */
public sealed interface Expression extends Statement {
  /**
   * A member access expression.
   *
   * @param object the expression being accessed
   * @param property the accessed identifier, integer, or (when {@code computed}) index/slice
   *     expression
   * @param computed whether {@code property} came from {@code [...]} rather than {@code .}
   * @param location the start of {@code object}
   */
  record MemberExpression(
      Expression object, Expression property, boolean computed, SourceLocation location)
      implements Expression {
    /** Rejects a null {@code object}, {@code property}, or {@code location}. */
    public MemberExpression {
      Objects.requireNonNull(object);
      Objects.requireNonNull(property);
      Objects.requireNonNull(location);
    }
  }

  /**
   * A call expression.
   *
   * @param callee the expression being called
   * @param args the call's positional, spread, and keyword arguments, in source order
   * @param location the start of {@code callee}
   */
  record CallExpression(Expression callee, List<Expression> args, SourceLocation location)
      implements Expression {
    /** Rejects a null {@code callee} or {@code location} and defensively copies {@code args}. */
    public CallExpression {
      Objects.requireNonNull(callee);
      args = List.copyOf(args);
      Objects.requireNonNull(location);
    }
  }

  /**
   * An identifier expression.
   *
   * @param value the identifier's name
   * @param location the start of the identifier
   */
  record Identifier(String value, SourceLocation location) implements Expression {
    /** Rejects a null {@code value} or {@code location}. */
    public Identifier {
      Objects.requireNonNull(value);
      Objects.requireNonNull(location);
    }
  }

  /**
   * An integer-syntax numeric literal.
   *
   * @param value the literal's numeric value
   * @param location the start of the literal
   */
  record IntegerLiteral(double value, SourceLocation location) implements Expression {
    /** Rejects a null {@code location}. */
    public IntegerLiteral {
      Objects.requireNonNull(location);
    }
  }

  /**
   * A decimal numeric literal.
   *
   * @param value the literal's numeric value
   * @param location the start of the literal
   */
  record FloatLiteral(double value, SourceLocation location) implements Expression {
    /** Rejects a null {@code location}. */
    public FloatLiteral {
      Objects.requireNonNull(location);
    }
  }

  /**
   * A string literal.
   *
   * @param value the literal's decoded text
   * @param location the start of the literal
   */
  record StringLiteral(String value, SourceLocation location) implements Expression {
    /** Rejects a null {@code value} or {@code location}. */
    public StringLiteral {
      Objects.requireNonNull(value);
      Objects.requireNonNull(location);
    }
  }

  /**
   * An array literal.
   *
   * @param value the literal's elements, in source order
   * @param location the start of the literal
   */
  record ArrayLiteral(List<Expression> value, SourceLocation location) implements Expression {
    /** Rejects a null {@code location} and defensively copies {@code value}. */
    public ArrayLiteral {
      value = List.copyOf(value);
      Objects.requireNonNull(location);
    }
  }

  /**
   * A tuple literal.
   *
   * @param value the literal's elements, in source order
   * @param location the start of the literal
   */
  record TupleLiteral(List<Expression> value, SourceLocation location) implements Expression {
    /** Rejects a null {@code location} and defensively copies {@code value}. */
    public TupleLiteral {
      value = List.copyOf(value);
      Objects.requireNonNull(location);
    }
  }

  /**
   * One object-literal entry.
   *
   * @param key the entry's key expression
   * @param value the entry's value expression
   */
  record ObjectEntry(Expression key, Expression value) {
    /** Rejects a null {@code key} or {@code value}. */
    public ObjectEntry {
      Objects.requireNonNull(key);
      Objects.requireNonNull(value);
    }
  }

  /**
   * An object literal retaining duplicate keys and insertion order.
   *
   * @param value the literal's entries, in source order
   * @param location the start of the literal
   */
  record ObjectLiteral(List<ObjectEntry> value, SourceLocation location) implements Expression {
    /** Rejects a null {@code location} and defensively copies {@code value}. */
    public ObjectLiteral {
      value = List.copyOf(value);
      Objects.requireNonNull(location);
    }
  }

  /**
   * A binary operator expression.
   *
   * @param operator the operator token
   * @param left the left-hand operand
   * @param right the right-hand operand
   * @param location the start of {@code left}
   */
  record BinaryExpression(
      Token operator, Expression left, Expression right, SourceLocation location)
      implements Expression {
    /** Rejects a null {@code operator}, {@code left}, {@code right}, or {@code location}. */
    public BinaryExpression {
      Objects.requireNonNull(operator);
      Objects.requireNonNull(left);
      Objects.requireNonNull(right);
      Objects.requireNonNull(location);
    }
  }

  /**
   * An expression transformed by a filter.
   *
   * @param operand the expression being filtered
   * @param filter the filter identifier or call
   * @param location the start of {@code operand}
   */
  record FilterExpression(Expression operand, Expression filter, SourceLocation location)
      implements Expression {
    /** Rejects a null {@code operand}, {@code filter}, or {@code location}. */
    public FilterExpression {
      Objects.requireNonNull(operand);
      Objects.requireNonNull(filter);
      Objects.requireNonNull(location);
    }
  }

  /**
   * A conditional select expression: {@code lhs if test}, with no {@code else} clause.
   *
   * @param lhs the expression yielded when {@code test} is truthy
   * @param test the select condition
   * @param location the start of {@code lhs}
   */
  record SelectExpression(Expression lhs, Expression test, SourceLocation location)
      implements Expression {
    /** Rejects a null {@code lhs}, {@code test}, or {@code location}. */
    public SelectExpression {
      Objects.requireNonNull(lhs);
      Objects.requireNonNull(test);
      Objects.requireNonNull(location);
    }
  }

  /**
   * An {@code is} test expression.
   *
   * @param operand the expression being tested
   * @param negate whether this is an {@code is not} test
   * @param test the named test being applied
   * @param location the start of {@code operand}
   */
  record TestExpression(
      Expression operand, boolean negate, Identifier test, SourceLocation location)
      implements Expression {
    /** Rejects a null {@code operand}, {@code test}, or {@code location}. */
    public TestExpression {
      Objects.requireNonNull(operand);
      Objects.requireNonNull(test);
      Objects.requireNonNull(location);
    }
  }

  /**
   * A unary operator expression.
   *
   * @param operator the operator token
   * @param argument the operand
   * @param location the start of {@code operator}
   */
  record UnaryExpression(Token operator, Expression argument, SourceLocation location)
      implements Expression {
    /** Rejects a null {@code operator}, {@code argument}, or {@code location}. */
    public UnaryExpression {
      Objects.requireNonNull(operator);
      Objects.requireNonNull(argument);
      Objects.requireNonNull(location);
    }
  }

  /**
   * A slice whose {@code start}, {@code stop}, and {@code step} components may each be null.
   *
   * @param start the slice's start bound, or null if omitted
   * @param stop the slice's stop bound, or null if omitted
   * @param step the slice's step, or null if omitted
   * @param location the start of the enclosing {@code [...]}
   */
  record SliceExpression(
      Expression start, Expression stop, Expression step, SourceLocation location)
      implements Expression {
    /**
     * Rejects a null {@code location}; {@code start}, {@code stop}, and {@code step} may be null.
     */
    public SliceExpression {
      Objects.requireNonNull(location);
    }
  }

  /**
   * A keyword call argument.
   *
   * @param key the argument's name
   * @param value the argument's value expression
   * @param location the start of {@code key}
   */
  record KeywordArgumentExpression(Identifier key, Expression value, SourceLocation location)
      implements Expression {
    /** Rejects a null {@code key}, {@code value}, or {@code location}. */
    public KeywordArgumentExpression {
      Objects.requireNonNull(key);
      Objects.requireNonNull(value);
      Objects.requireNonNull(location);
    }
  }

  /**
   * A spread call argument.
   *
   * @param argument the expression being spread
   * @param location the start of the {@code *} operator
   */
  record SpreadExpression(Expression argument, SourceLocation location) implements Expression {
    /** Rejects a null {@code argument} or {@code location}. */
    public SpreadExpression {
      Objects.requireNonNull(argument);
      Objects.requireNonNull(location);
    }
  }

  /**
   * A three-branch conditional expression.
   *
   * @param condition the ternary's condition
   * @param trueExpr the value yielded when {@code condition} is truthy
   * @param falseExpr the value yielded otherwise
   * @param location the start of {@code trueExpr}
   */
  record Ternary(
      Expression condition, Expression trueExpr, Expression falseExpr, SourceLocation location)
      implements Expression {
    /**
     * Rejects a null {@code condition}, {@code trueExpr}, {@code falseExpr}, or {@code location}.
     */
    public Ternary {
      Objects.requireNonNull(condition);
      Objects.requireNonNull(trueExpr);
      Objects.requireNonNull(falseExpr);
      Objects.requireNonNull(location);
    }
  }
}
